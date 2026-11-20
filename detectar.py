"""Linha de comando: analisa fotos, pastas, vídeos ou a webcam.

Exemplos:
    python detectar.py --imagem exemplos/victory.jpg --mostrar
    python detectar.py --pasta minhas_fotos --borrar
    python detectar.py --video festa.mp4
    python detectar.py --webcam 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src import DetectorGestos

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analisar_imagem(det: DetectorGestos, caminho: Path, saida: Path, borrar: bool, mostrar: bool):
    img = cv2.imread(str(caminho))
    if img is None:
        print(f"[!] Nao foi possivel abrir {caminho}")
        return None
    t0 = time.perf_counter()
    res = det.analisar(img)
    ms = (time.perf_counter() - t0) * 1000

    print(f"\n== {caminho.name}  ({ms:.0f} ms, {len(res.pessoas)} pessoa(s), {len(res.maos)} mao(s))")
    for i, m in enumerate(res.maos, 1):
        print(f"   mao {i}: {m.gesto.nome:<20} dedos esticados={m.dedos_esticados}")
    print("   " + res.resumo().replace("\n", "\n   "))

    saida.mkdir(parents=True, exist_ok=True)
    anotada = det.desenhar(img, res)
    cv2.imwrite(str(saida / f"{caminho.stem}_analise.jpg"), anotada)
    if borrar and res.alerta:
        cv2.imwrite(str(saida / f"{caminho.stem}_segura.jpg"), det.borrar_gestos(img, res))
    if mostrar:
        _mostrar(anotada, caminho.name)
        cv2.waitKey(0)
    return res


def analisar_fluxo(det: DetectorGestos, fonte, saida: Path, borrar: bool):
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        raise SystemExit(f"Nao foi possivel abrir a fonte de video: {fonte}")

    gravador = None
    if isinstance(fonte, str):  # arquivo de vídeo -> salva o resultado
        saida.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        destino = saida / f"{Path(fonte).stem}_analise.mp4"
        gravador = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Gravando resultado em {destino}")

    print("Teclas: [q] sair  [s] salvar quadro  [b] alternar borrar gestos")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if isinstance(fonte, int):
            frame = cv2.flip(frame, 1)  # espelha a webcam, como um "selfie"
        t0 = time.perf_counter()
        res = det.analisar(frame)
        fps = 1 / max(time.perf_counter() - t0, 1e-6)

        base = det.borrar_gestos(frame, res) if borrar else frame
        anotada = det.desenhar(base, res)
        cv2.putText(anotada, f"{fps:.1f} FPS", (10, anotada.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if gravador:
            gravador.write(anotada)
        _mostrar(anotada, "Detector de gestos")
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("b"):
            borrar = not borrar
        if tecla == ord("s"):
            saida.mkdir(parents=True, exist_ok=True)
            n += 1
            cv2.imwrite(str(saida / f"quadro_{n:03d}.jpg"), anotada)
            print(f"Quadro salvo: quadro_{n:03d}.jpg | {res.resumo()}")

    cap.release()
    if gravador:
        gravador.release()
    cv2.destroyAllWindows()


def _mostrar(img, titulo):
    h, w = img.shape[:2]
    escala = min(1.0, 900 / max(h, w))
    cv2.imshow(titulo, cv2.resize(img, None, fx=escala, fy=escala) if escala < 1 else img)


def main():
    ap = argparse.ArgumentParser(description="Detecta gestos associados a faccoes em fotos, videos ou webcam.")
    fonte = ap.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--imagem", type=Path, help="caminho de uma foto")
    fonte.add_argument("--pasta", type=Path, help="pasta com varias fotos")
    fonte.add_argument("--video", type=str, help="arquivo de video")
    fonte.add_argument("--webcam", type=int, help="indice da webcam (normalmente 0)")
    ap.add_argument("--saida", type=Path, default=Path("resultados"), help="pasta de saida (padrao: resultados)")
    ap.add_argument("--borrar", action="store_true", help="gera/mostra versao com os gestos de risco pixelados")
    ap.add_argument("--mostrar", action="store_true", help="abre janela com o resultado (imagens)")
    ap.add_argument("--sem-yolo", action="store_true", help="nao usa YOLO (mais rapido, pior para fotos de grupo)")
    args = ap.parse_args()

    # Na webcam a pessoa costuma estar perto da câmera; sem YOLO fica mais fluido.
    usar_yolo = not args.sem_yolo and args.webcam is None
    det = DetectorGestos(usar_yolo=usar_yolo)

    if args.imagem:
        analisar_imagem(det, args.imagem, args.saida, args.borrar, args.mostrar)
    elif args.pasta:
        fotos = sorted(p for p in args.pasta.rglob("*") if p.suffix.lower() in EXTENSOES)
        alertas = sum(bool(r and r.alerta) for r in (analisar_imagem(det, f, args.saida, args.borrar, args.mostrar) for f in fotos))
        print(f"\nTotal: {len(fotos)} foto(s), {alertas} com alerta. Resultados em {args.saida}/")
    else:
        analisar_fluxo(det, args.video if args.video else args.webcam, args.saida, args.borrar)


if __name__ == "__main__":
    main()
