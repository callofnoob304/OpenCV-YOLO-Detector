"""Classificação de gestos a partir dos 21 pontos (landmarks) da mão.

Pontos do MediaPipe Hands:
    0 = punho
    1-4   = polegar   (CMC, MCP, IP, ponta)
    5-8   = indicador (MCP, PIP, DIP, ponta)
    9-12  = médio
    13-16 = anelar
    17-20 = mínimo

A classificação é feita por regras geométricas (ângulos entre articulações),
usando as coordenadas 3D "world" (em metros), que não dependem da posição,
escala ou rotação da mão na imagem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (MCP, PIP, DIP, TIP) de cada dedo longo
DEDOS = {
    "indicador": (5, 6, 7, 8),
    "medio": (9, 10, 11, 12),
    "anelar": (13, 14, 15, 16),
    "minimo": (17, 18, 19, 20),
}
NOMES_DEDOS = ["polegar", "indicador", "medio", "anelar", "minimo"]

# Limiares (graus). Um dedo é "esticado" quando suas articulações estão quase retas.
ANGULO_DEDO_ESTICADO = 150.0   # soma média PIP/DIP
ANGULO_POLEGAR_ESTICADO = 145.0
DIST_POLEGAR_ABERTO = 0.75      # distância ponta do polegar -> MCP do indicador / tamanho da palma


@dataclass
class Gesto:
    codigo: str          # identificador curto (ex.: "V", "TRES", "NEUTRO")
    nome: str            # nome amigável
    risco: bool          # se o gesto é associado a facções
    descricao: str       # explicação mostrada ao usuário


GESTOS = {
    "V": Gesto(
        "V", "Dois dedos (V)", True,
        "Gesto 'V' / 'paz e amor'. Pode ser interpretado como 'tudo dois', "
        "sinal associado ao Comando Vermelho (CV).",
    ),
    "TRES": Gesto(
        "TRES", "Tres dedos", True,
        "Gesto de tres dedos. Pode ser interpretado como 'tudo tres', "
        "sinal associado ao PCC e grupos aliados.",
    ),
    "NEUTRO": Gesto("NEUTRO", "Sem gesto de risco", False, "Nenhum gesto de risco identificado."),
}


def _angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo (graus) no ponto b formado por a-b-c. 180 = totalmente reto."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def estado_dedos(pts: np.ndarray) -> list[bool]:
    """Retorna [polegar, indicador, medio, anelar, minimo] -> True se esticado.

    pts: array (21, 3) com as coordenadas da mão.
    """
    estados = []

    # Polegar: articulação IP reta E ponta afastada da base do indicador
    ang_pol = (_angulo(pts[1], pts[2], pts[3]) + _angulo(pts[2], pts[3], pts[4])) / 2
    palma = np.linalg.norm(pts[0] - pts[9]) + 1e-9
    dist_pol = np.linalg.norm(pts[4] - pts[5]) / palma
    estados.append(ang_pol > ANGULO_POLEGAR_ESTICADO and dist_pol > DIST_POLEGAR_ABERTO)

    # Dedos longos: ângulos em PIP e DIP quase retos E ponta mais longe do punho que o PIP
    for mcp, pip, dip, tip in DEDOS.values():
        ang = (_angulo(pts[mcp], pts[pip], pts[dip]) + _angulo(pts[pip], pts[dip], pts[tip])) / 2
        longe = np.linalg.norm(pts[tip] - pts[0]) > np.linalg.norm(pts[pip] - pts[0])
        estados.append(ang > ANGULO_DEDO_ESTICADO and longe)

    return estados


def classificar(estados: list[bool]) -> Gesto:
    """Mapeia o padrão de dedos esticados para um gesto."""
    pol, ind, med, ane, mini = estados

    # Três dedos: indicador+médio+anelar (polegar e mínimo dobrados)
    if ind and med and ane and not mini and not pol:
        return GESTOS["TRES"]
    # Três dedos (variação comum no Brasil): polegar+indicador+médio
    if pol and ind and med and not ane and not mini:
        return GESTOS["TRES"]
    # V: apenas indicador e médio
    if ind and med and not ane and not mini and not pol:
        return GESTOS["V"]
    return GESTOS["NEUTRO"]
