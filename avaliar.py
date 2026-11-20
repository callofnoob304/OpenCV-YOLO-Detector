"""Avalia o detector em um conjunto de fotos rotuladas e gera métricas.

Organize as fotos assim (o nome da pasta é o rótulo verdadeiro):
    dataset/
        V/        fotos com o gesto "V" (dois dedos)
        TRES/     fotos com o gesto de três dedos
        NEUTRO/   fotos sem gesto de risco (mãos abertas, joinha, sem mãos...)

Executar:  python avaliar.py --dataset dataset
Saída: matriz de confusão, acurácia, precisão/recall do alerta e resultados/avaliacao.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from src import DetectorGestos

CLASSES = ["V", "TRES", "NEUTRO"]
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def previsao_da_foto(res) -> str:
    codigos = {m.gesto.codigo for m in res.maos_de_risco}
    if "TRES" in codigos:
        return "TRES"
    if "V" in codigos:
        return "V"
    return "NEUTRO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--saida", type=Path, default=Path("resultados"))
    ap.add_argument("--sem-yolo", action="store_true")
    args = ap.parse_args()

    det = DetectorGestos(usar_yolo=not args.sem_yolo)
    matriz = {r: {p: 0 for p in CLASSES} for r in CLASSES}
    linhas, tempos = [], []

    for real in CLASSES:
        pasta = args.dataset / real
        if not pasta.exists():
            continue
        for foto in sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES):
            img = cv2.imread(str(foto))
            if img is None:
                continue
            t0 = time.perf_counter()
            res = det.analisar(img)
            tempos.append(time.perf_counter() - t0)
            prev = previsao_da_foto(res)
            matriz[real][prev] += 1
            linhas.append([str(foto), real, prev, len(res.maos), "ok" if prev == real else "ERRO"])

    total = sum(sum(l.values()) for l in matriz.values())
    if total == 0:
        raise SystemExit(f"Nenhuma foto encontrada em {args.dataset}/(V|TRES|NEUTRO)")

    print("\nMatriz de confusao (linhas = real, colunas = previsto)")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES))
    for r in CLASSES:
        print(f"{r:>8}" + "".join(f"{matriz[r][p]:>8}" for p in CLASSES))

    acertos = sum(matriz[c][c] for c in CLASSES)
    print(f"\nAcuracia (3 classes): {acertos}/{total} = {acertos / total:.1%}")

    # Visão binária: "deveria alertar?" x "alertou?"
    risco = ["V", "TRES"]
    vp = sum(matriz[r][p] for r in risco for p in risco)
    fn = sum(matriz[r]["NEUTRO"] for r in risco)
    fp = sum(matriz["NEUTRO"][p] for p in risco)
    precisao = vp / (vp + fp) if vp + fp else 0
    recall = vp / (vp + fn) if vp + fn else 0
    print(f"Alerta  -> precisao: {precisao:.1%}  recall: {recall:.1%}  (VP={vp} FP={fp} FN={fn})")
    print(f"Tempo medio por foto: {1000 * sum(tempos) / len(tempos):.0f} ms")

    args.saida.mkdir(parents=True, exist_ok=True)
    with open(args.saida / "avaliacao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "real", "previsto", "maos", "status"])
        w.writerows(linhas)
    print(f"Detalhes por foto: {args.saida / 'avaliacao.csv'}")


if __name__ == "__main__":
    main()
