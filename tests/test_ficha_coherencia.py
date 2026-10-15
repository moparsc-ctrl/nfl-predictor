"""Coherencia interna de la ficha: marcador/total/barra/recomendacion.

Correr con:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.market import blend_prediction, marcadores  # noqa: E402
from model.poisson_model import (  # noqa: E402
    EDGE_ALTO, UMBRAL_VALOR, recomendacion_apuesta,
)

# (local, visitante, modelo, spread_line, total_line)
CASOS = {
    # modelo pone al local ligeramente arriba, mercado favorece al visitante (SEA @ DEN)
    "DEN_SEA": ("DEN", "SEA",
                {"lambda_home": 18.8, "lambda_away": 18.6, "p_home_win": 55.0,
                 "p_away_win": 45.0, "total_esperado_puntos": 37.4}, -3.0, 41.5),
    # modelo ve visitante muy arriba, mercado favorece al local (BUF @ LA)
    "LA_BUF": ("LA", "BUF",
               {"lambda_home": 28.3, "lambda_away": 31.5, "p_home_win": 37.0,
                "p_away_win": 63.0, "total_esperado_puntos": 59.8}, 3.0, 51.5),
    "PAREJO": ("AAA", "BBB",
               {"lambda_home": 22.0, "lambda_away": 22.0, "p_home_win": 52.0,
                "p_away_win": 48.0, "total_esperado_puntos": 44.0}, 0.0, 44.0),
    "FUERTE": ("AAA", "BBB",
               {"lambda_home": 30.0, "lambda_away": 15.0, "p_home_win": 90.0,
                "p_away_win": 10.0, "total_esperado_puntos": 45.0}, 10.0, 44.0),
}


def _pred(k):
    h, a, m, sp, tl = CASOS[k]
    return h, a, blend_prediction(dict(m), sp, tl)


@pytest.mark.parametrize("k", CASOS)
def test_a_total_mostrado_es_suma_del_marcador_mostrado(k):
    _, _, pred = _pred(k)
    for f in marcadores(pred):
        assert f["total"] == pytest.approx(f["home"] + f["away"], abs=1e-9)
        assert f["total_etiqueta"]  # nunca un total sin etiqueta


@pytest.mark.parametrize("k", CASOS)
def test_b_ganador_del_marcador_coincide_con_su_probabilidad(k):
    _, _, pred = _pred(k)
    for f in marcadores(pred):
        if f["home"] == f["away"]:
            continue
        assert (f["home"] > f["away"]) == (f["p_home"] > 50), f


def test_b_marcador_ajustado_coincide_con_la_barra_en_den_sea():
    h, a, pred = _pred("DEN_SEA")
    aj = [f for f in marcadores(pred) if f["clave"] == "ajustado"][0]
    assert pred["p_away_win"] > pred["p_home_win"]  # barra: visitante
    assert aj["away"] > aj["home"]
    # el marcador puro, en cambio, puede discrepar: por eso esta etiquetado
    puro = [f for f in marcadores(pred) if f["clave"] == "puro"][0]
    assert "PURO" in puro["etiqueta"]


def test_ajustado_suma_el_total_mezclado():
    _, _, pred = _pred("DEN_SEA")
    aj = [f for f in marcadores(pred) if f["clave"] == "ajustado"][0]
    assert aj["total"] == pytest.approx(pred["total_esperado_puntos"], abs=0.11)


@pytest.mark.parametrize("k", CASOS)
def test_c_valor_apunta_al_equipo_con_edge_positivo(k):
    h, a, pred = _pred(k)
    rec = recomendacion_apuesta(pred, h, a)
    if rec["valor_lado"] is None:
        assert max(pred["edge_home"], pred["edge_away"]) < UMBRAL_VALOR
        assert rec["valor_texto"].startswith("Sin diferencia")
        return
    edge = pred["edge_home"] if rec["valor_lado"] == "home" else pred["edge_away"]
    assert edge >= UMBRAL_VALOR
    equipo = h if rec["valor_lado"] == "home" else a
    assert equipo in rec["valor_texto"]
    otro = a if equipo == h else h
    assert otro not in rec["valor_texto"]


def test_c_favorito_y_valor_pueden_diferir_pero_estan_separados():
    h, a, pred = _pred("DEN_SEA")
    rec = recomendacion_apuesta(pred, h, a)
    assert rec["favorito_lado"] == "away"      # SEA favorito
    assert rec["valor_lado"] == "home"         # valor en DEN
    assert "SEA" in rec["favorito_texto"] and "DEN" in rec["valor_texto"]


def test_alerta_edge_muy_grande():
    h, a, pred = _pred("LA_BUF")
    rec = recomendacion_apuesta(pred, h, a)
    assert pred["edge_away"] >= EDGE_ALTO
    assert rec["alerta_texto"]


def test_sin_linea_no_hay_marcador_ajustado_ni_valor():
    m = dict(CASOS["DEN_SEA"][2])
    pred = blend_prediction(m, None, None)
    assert [f["clave"] for f in marcadores(pred)] == ["puro"]
    rec = recomendacion_apuesta(pred, "DEN", "SEA")
    assert rec["valor_lado"] is None
