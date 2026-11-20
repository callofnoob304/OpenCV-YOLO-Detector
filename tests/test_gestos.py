"""Testes das regras de classificação (não precisam de modelos nem imagens).

Executar:  python -m pytest tests
"""

import numpy as np
import pytest

from src.gestos import classificar, estado_dedos

# Base de cada dedo (MCP) numa mão sintética apontando para cima (eixo y negativo)
BASES = {1: (-0.03, -0.01), 5: (-0.02, -0.08), 9: (0.0, -0.085), 13: (0.02, -0.08), 17: (0.04, -0.07)}


def mao_sintetica(esticados):
    """Cria 21 pontos 3D de uma mão com os dedos [polegar..minimo] esticados ou dobrados."""
    pts = np.zeros((21, 3))
    # polegar: 1..4
    if esticados[0]:
        for i, k in enumerate(range(1, 5)):
            pts[k] = (-0.03 - 0.025 * i, -0.01 - 0.015 * i, 0)
    else:  # dobrado por cima da palma
        pts[1], pts[2], pts[3], pts[4] = (-0.03, -0.01, 0), (-0.045, -0.035, 0), (-0.02, -0.05, 0), (0.0, -0.045, 0)
    # dedos longos
    for d, base in enumerate([5, 9, 13, 17], start=1):
        bx, by = BASES[base]
        pts[base] = (bx, by, 0)
        if esticados[d]:
            for j in range(1, 4):
                pts[base + j] = (bx, by - 0.025 * j, 0)
        else:  # curvado para dentro da palma
            pts[base + 1] = (bx, by - 0.025, -0.01)
            pts[base + 2] = (bx, by - 0.01, -0.03)
            pts[base + 3] = (bx, by + 0.01, -0.02)
    return pts


@pytest.mark.parametrize(
    "dedos, esperado",
    [
        ([0, 1, 1, 0, 0], "V"),
        ([0, 1, 1, 1, 0], "TRES"),
        ([1, 1, 1, 0, 0], "TRES"),
        ([1, 1, 1, 1, 1], "NEUTRO"),   # mão aberta
        ([0, 0, 0, 0, 0], "NEUTRO"),   # punho fechado
        ([1, 0, 0, 0, 0], "NEUTRO"),   # joinha
        ([0, 1, 0, 0, 0], "NEUTRO"),   # apontando
        ([0, 1, 0, 0, 1], "NEUTRO"),   # "rock"
    ],
)
def test_classificacao(dedos, esperado):
    pts = mao_sintetica([bool(d) for d in dedos])
    assert estado_dedos(pts) == [bool(d) for d in dedos]
    assert classificar(estado_dedos(pts)).codigo == esperado
