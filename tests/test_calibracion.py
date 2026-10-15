"""Calibracion: probabilidad de victoria = Normal sobre el margen (sigma configurable).

Correr con:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import config as cfg  # noqa: E402
from model.market import (  # noqa: E402
    analizar_spread_total, blend_prediction, marcadores, prob_cubrir_spread, prob_over,
)
from model.poisson_model import predict_match, prob_victoria_normal  # noqa: E402

LG = 22.0


def _df():
    row = {"off_strength": 1.0, "def_strength": 1.0, "off_strength_epa": 1.0, "def_strength_epa": 1.0}
    return pd.DataFrame({"AAA": row, "BBB": row}).T


def _pred(margen, **kw):
    """Partido con margen esperado exacto = `margen` (via home_field_boost, sin sesgo de total)."""
    return predict_match("AAA", "BBB", _df(), LG, home_field_boost=1 + margen / LG,
                         total_bias=0.0, **kw)


# ---- (a) margen 0 -> 50% +-0.5
def test_a_margen_cero_da_50():
    assert prob_victoria_normal(0.0) * 100 == pytest.approx(50.0, abs=0.5)
    p = _pred(0.0)
    assert p["p_home_win"] == pytest.approx(50.0, abs=0.5)


# ---- (b) margen 3 con sigma 13.5 -> entre 58% y 60%
def test_b_margen_3_sigma_13_5():
    assert 58.0 <= prob_victoria_normal(3.0, 13.5) * 100 <= 60.0
    p = _pred(3.0, sigma_margen=13.5)
    assert 58.0 <= p["p_home_win"] <= 60.0
    # y con la sigma por defecto (13.6)
    assert 58.0 <= _pred(3.0)["p_home_win"] <= 60.0


# ---- (c) los dos equipos suman 100%
@pytest.mark.parametrize("margen", [-14, -7, -3, 0, 0.2, 3, 7, 14, 21])
def test_c_probabilidades_suman_100(margen):
    p = _pred(margen)
    assert p["p_home_win"] + p["p_away_win"] == pytest.approx(100.0, abs=0.11)
    for m in ("normal", "poisson"):
        q = _pred(margen, method=m)
        assert q["p_home_win"] + q["p_away_win"] == pytest.approx(100.0, abs=0.11)


def test_sigma_es_parametro_no_fijo():
    estrecho = _pred(3.0, sigma_margen=6.0)["p_home_win"]
    ancho = _pred(3.0, sigma_margen=20.0)["p_home_win"]
    assert estrecho > _pred(3.0)["p_home_win"] > ancho > 50.0


def test_normal_es_menos_confiada_que_poisson():
    # Poisson con ~45 pts totales tiene sigma ~6.7: mucho mas extremo que la normal
    pn = _pred(3.0, method="normal")["p_home_win"]
    pp = _pred(3.0, method="poisson")["p_home_win"]
    assert pp > pn + 5


# ---- sesgo del total
def test_total_bias_cambia_el_total_pero_no_el_margen_ni_la_suma_mostrada():
    base = predict_match("AAA", "BBB", _df(), LG, total_bias=0.0)
    corr = predict_match("AAA", "BBB", _df(), LG, total_bias=-1.0)
    assert corr["total_esperado_puntos"] == pytest.approx(base["total_esperado_puntos"] - 1.0, abs=0.11)
    assert corr["margen_esperado"] == pytest.approx(base["margen_esperado"], abs=0.01)
    fila = marcadores(blend_prediction(dict(corr), None, None))[0]
    assert fila["total"] == pytest.approx(fila["home"] + fila["away"])


def test_defaults_vienen_de_config():
    assert cfg.SIGMA_MARGEN_MODELO == 13.6 and cfg.SIGMA_TOTAL == 13.9
    p = predict_match("AAA", "BBB", _df(), LG)
    assert p["sigma_margen"] == cfg.SIGMA_MARGEN_MODELO
    assert p["metodo"].startswith("Normal")


# ---- peso de la mezcla configurable
def test_peso_de_mezcla_configurable():
    m = _pred(7.0)
    soo_mercado = blend_prediction(dict(m), -3.0, 44.0, market_weight=1.0)
    assert soo_mercado["p_home_win"] == pytest.approx(soo_mercado["market_home_prob"], abs=0.11)
    mezcla = blend_prediction(dict(m), -3.0, 44.0, market_weight=0.65)
    assert mezcla["p_home_win"] != soo_mercado["p_home_win"]
    por_defecto = blend_prediction(dict(m), -3.0, 44.0)
    assert por_defecto["market_weight"] == cfg.MARKET_WEIGHT


# ---- cobertura de spread y over/under
def test_cobertura_spread_y_over_suman_100():
    h, a, push = prob_cubrir_spread(3.0, 2.5)
    assert h + a + push == pytest.approx(1.0)
    o, u = prob_over(45.0, 44.0)
    assert o + u == pytest.approx(1.0)
    assert o > 0.5 > u
    # modelo espera margen 3 y la linea es 2.5 -> local cubre >50%
    assert h > 0.5


def test_numeros_clave_opcional_suma_1():
    h, a, push = prob_cubrir_spread(3.0, 3.0, numeros_clave=True)
    assert h + a + push == pytest.approx(1.0)
    assert push > 0.05  # el 3 pesa mucho: hay masa real de push exacto en 3


def test_analisis_spread_total_en_ficha():
    m = _pred(3.0)
    pred = blend_prediction(dict(m), 2.5, 44.5)
    r = analizar_spread_total(pred, {"over": -105}, {"total_linea": 44.5})
    assert [f["clave"] for f in r["filas"]] == ["sp_home", "sp_away", "over", "under"]
    f = {x["clave"]: x for x in r["filas"]}
    assert f["sp_home"]["prob"] + f["sp_away"]["prob"] + f["sp_home"]["p_push"] == pytest.approx(100.0, abs=0.2)
    assert f["over"]["prob"] + f["under"]["prob"] + f["over"]["p_push"] == pytest.approx(100.0, abs=0.2)
    assert f["over"]["cuota"] == -105 and not f["over"]["cuota_estandar"]   # tu cuota sobre TU total
    assert f["under"]["cuota"] == -110 and f["under"]["cuota_estandar"]
    assert analizar_spread_total(blend_prediction(dict(m), None, None)) is None
