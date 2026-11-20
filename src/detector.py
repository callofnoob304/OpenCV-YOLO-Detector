"""Pipeline de detecção: YOLO (pessoas) -> MediaPipe (mãos) -> regras (gestos) -> OpenCV (saída).

1. O YOLO localiza as pessoas na foto. Cada pessoa é recortada e ampliada,
   o que permite achar mãos pequenas em fotos de grupo ou tiradas de longe.
2. O MediaPipe HandLandmarker encontra as mãos (na imagem inteira e em cada
   recorte) e devolve 21 pontos 3D por mão.
3. As regras de `gestos.py` decidem quais dedos estão esticados e se o
   padrão corresponde a um gesto de risco ("V" ou "três dedos").
4. O OpenCV desenha o resultado, gera o alerta e, se o usuário quiser,
   borra as mãos com gesto de risco.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Reduz as mensagens de log internas do MediaPipe/TensorFlow Lite
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .gestos import Gesto, NOMES_DEDOS, classificar, estado_dedos

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "models"
URL_HAND_LANDMARKER = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Ligações entre os pontos da mão (para desenhar o "esqueleto")
CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

VERMELHO = (40, 40, 230)
VERDE = (80, 200, 80)
BRANCO = (255, 255, 255)


@dataclass
class Mao:
    bbox: tuple[int, int, int, int]       # x1, y1, x2, y2 em pixels
    pontos: np.ndarray                    # (21, 2) em pixels
    estados: list[bool]                   # dedos esticados
    gesto: Gesto
    lado: str                             # "Left" / "Right"
    confianca: float

    @property
    def dedos_esticados(self) -> list[str]:
        return [n for n, e in zip(NOMES_DEDOS, self.estados) if e]


@dataclass
class Resultado:
    maos: list[Mao] = field(default_factory=list)
    pessoas: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def maos_de_risco(self) -> list[Mao]:
        return [m for m in self.maos if m.gesto.risco]

    @property
    def alerta(self) -> bool:
        return bool(self.maos_de_risco)

    def resumo(self) -> str:
        if not self.alerta:
            return f"OK - {len(self.maos)} mao(s) analisada(s), nenhum gesto de risco."
        linhas = [f"ALERTA - {len(self.maos_de_risco)} gesto(s) de risco encontrado(s):"]
        for m in self.maos_de_risco:
            linhas.append(f"  * {m.gesto.nome}: {m.gesto.descricao}")
        return "\n".join(linhas)


def _baixar_modelo_mao() -> Path:
    destino = PASTA_MODELOS / "hand_landmarker.task"
    if not destino.exists():
        PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
        print(f"Baixando modelo de maos do MediaPipe para {destino} ...")
        urllib.request.urlretrieve(URL_HAND_LANDMARKER, destino)
    return destino


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


class DetectorGestos:
    def __init__(
        self,
        usar_yolo: bool = True,
        modelo_yolo: str = "yolo11n.pt",
        conf_pessoa: float = 0.4,
        conf_mao: float = 0.5,
        max_maos: int = 6,
    ):
        self.usar_yolo = usar_yolo
        self.conf_pessoa = conf_pessoa
        self.yolo = None
        if usar_yolo:
            from ultralytics import YOLO  # import aqui para o modo sem YOLO iniciar rápido

            caminho = PASTA_MODELOS / modelo_yolo
            PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
            self.yolo = YOLO(str(caminho))  # baixa automaticamente na 1a execucao

        opcoes = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_baixar_modelo_mao())),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_maos,
            min_hand_detection_confidence=conf_mao,
            min_hand_presence_confidence=conf_mao,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opcoes)

    # ------------------------------------------------------------------ etapas
    def detectar_pessoas(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.yolo is None:
            return []
        res = self.yolo.predict(img_bgr, classes=[0], conf=self.conf_pessoa, verbose=False)[0]
        return [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]

    def _maos_na_regiao(self, img_bgr, x0, y0, x1, y1) -> list[Mao]:
        """Roda o MediaPipe em uma região da imagem e devolve mãos em coordenadas globais."""
        recorte = img_bgr[y0:y1, x0:x1]
        if recorte.size == 0:
            return []
        h, w = recorte.shape[:2]
        # Amplia recortes pequenos: o detector de mãos funciona melhor com ~640 px
        escala = max(1.0, 640 / max(h, w))
        if escala > 1.0:
            recorte = cv2.resize(recorte, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
        rgb = np.ascontiguousarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        maos = []
        for lms, world, lado in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            px = np.array([[x0 + p.x * w, y0 + p.y * h] for p in lms])
            pts3d = np.array([[p.x, p.y, p.z] for p in world])
            estados = estado_dedos(pts3d)
            (bx1, by1), (bx2, by2) = px.min(0), px.max(0)
            margem = 0.15 * max(bx2 - bx1, by2 - by1)
            bbox = (int(bx1 - margem), int(by1 - margem), int(bx2 + margem), int(by2 + margem))
            maos.append(Mao(bbox, px, estados, classificar(estados), lado[0].category_name, lado[0].score))
        return maos

    # ------------------------------------------------------------------ API
    def analisar(self, img_bgr: np.ndarray) -> Resultado:
        H, W = img_bgr.shape[:2]
        pessoas = self.detectar_pessoas(img_bgr)

        candidatas = self._maos_na_regiao(img_bgr, 0, 0, W, H)
        for (x1, y1, x2, y2) in pessoas:
            # recorte com margem, para não cortar mãos levantadas
            mx, my = int(0.2 * (x2 - x1)), int(0.15 * (y2 - y1))
            rx1, ry1 = max(0, x1 - mx), max(0, y1 - my)
            rx2, ry2 = min(W, x2 + mx), min(H, y2 + my)
            if (rx2 - rx1) * (ry2 - ry1) > 0.8 * W * H:
                continue  # recorte ≈ imagem inteira, já analisada
            candidatas += self._maos_na_regiao(img_bgr, rx1, ry1, rx2, ry2)

        # Remove mãos duplicadas (a mesma mão achada na imagem inteira e no recorte).
        # Em caso de conflito, dá prioridade a quem detectou gesto de risco.
        candidatas.sort(key=lambda m: (m.gesto.risco, m.confianca), reverse=True)
        maos: list[Mao] = []
        for m in candidatas:
            if all(_iou(m.bbox, k.bbox) < 0.3 for k in maos):
                maos.append(m)
        return Resultado(maos=maos, pessoas=pessoas)

    # ------------------------------------------------------------------ saída
    @staticmethod
    def desenhar(img_bgr: np.ndarray, resultado: Resultado, mostrar_pessoas: bool = True) -> np.ndarray:
        out = img_bgr.copy()
        H, W = out.shape[:2]
        esp = max(1, round(max(H, W) / 600))

        if mostrar_pessoas:
            for (x1, y1, x2, y2) in resultado.pessoas:
                cv2.rectangle(out, (x1, y1), (x2, y2), (200, 160, 60), esp)

        for m in resultado.maos:
            cor = VERMELHO if m.gesto.risco else VERDE
            for a, b in CONEXOES:
                cv2.line(out, tuple(m.pontos[a].astype(int)), tuple(m.pontos[b].astype(int)), BRANCO, esp)
            for p in m.pontos:
                cv2.circle(out, tuple(p.astype(int)), esp + 2, cor, -1)
            x1, y1, x2, y2 = m.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), cor, esp + 1)
            _rotulo(out, m.gesto.nome, (x1, y1), cor, esp)

        # Faixa de alerta no topo
        faixa = max(36, H // 14)
        cv2.rectangle(out, (0, 0), (W, faixa), VERMELHO if resultado.alerta else VERDE, -1)
        texto = (
            f"ATENCAO: {len(resultado.maos_de_risco)} gesto(s) associado(s) a faccoes"
            if resultado.alerta else "Nenhum gesto de risco detectado"
        )
        # ajusta o tamanho da fonte para caber na largura da imagem
        escala = faixa / 55
        largura = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, esp + 1)[0][0]
        if largura > W - 24:
            escala *= (W - 24) / largura
        cv2.putText(out, texto, (12, int(faixa * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, escala, BRANCO, esp + 1, cv2.LINE_AA)
        return out

    @staticmethod
    def borrar_gestos(img_bgr: np.ndarray, resultado: Resultado) -> np.ndarray:
        """Gera uma versão 'segura' da foto com as mãos de risco pixeladas."""
        out = img_bgr.copy()
        H, W = out.shape[:2]
        for m in resultado.maos_de_risco:
            x1, y1, x2, y2 = m.bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            reg = out[y1:y2, x1:x2]
            if reg.size == 0:
                continue
            peq = cv2.resize(reg, (max(1, reg.shape[1] // 14), max(1, reg.shape[0] // 14)), interpolation=cv2.INTER_LINEAR)
            out[y1:y2, x1:x2] = cv2.resize(peq, (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_NEAREST)
        return out


def _rotulo(img, texto, origem, cor, esp):
    escala = 0.5 * esp
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, esp)
    x, y = origem
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), cor, -1)
    cv2.putText(img, texto, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, escala, BRANCO, esp, cv2.LINE_AA)
