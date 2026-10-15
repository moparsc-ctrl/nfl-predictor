"""Comparacion de lineas: tu casa vs consenso, con frecuencias historicas."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.lineas import (  # noqa: E402
    cargar_historico, comparar_casa, distribucion_condicionada, equilibrar, ev_100, momio_justo,
)

HIST = cargar_historico()
TODAS = {"sp_home": -110, "sp_away": -110, "over": -110, "under": -110, "ml_home": 120, "ml_away": -143}


def _f(r):
    return {f["clave"]: f for f in r["filas"]}


def test_historico_cargado_2015_2025():
    assert len(HIST) > 2800
    assert {r[0] for r in HIST} >= {2015, 2025}


def test_pmf_suma_1_y_ventana_con_muestra_suficiente():
    pmf, n, h = distribucion_condicionada(HIST, 3.0, 2, 4, 1.0)
    assert sum(pmf.values()) == pytest.approx(1.0)
    assert n >= 200


def test_el_consenso_es_50_50_en_su_propia_linea():
    r = comparar_casa(-2.5, 40.5, TODAS)
    f = _f(r)
    assert f["sp_home"]["p_gana"] == pytest.approx(50.0, abs=0.01)
    assert f["over"]["p_gana"] == pytest.approx(50.0, abs=0.01)
    # con -110 a cada lado y p=50% el VE es -4.55 (el margen de la casa)
    assert f["sp_home"]["ev"] == pytest.approx(-4.55, abs=0.01)


def test_empate_en_3_y_7_sale_de_frecuencia_historica():
    # historicamente el margen cae exacto en 3 o 7 mucho mas que en una normal (~2.5-3%)
    r3 = comparar_casa(3.0, 44.0, TODAS)
    r7 = comparar_casa(7.0, 44.5, TODAS)
    assert r3["push_spread"] > 7.0 and r7["push_spread"] > 5.0
    # linea de medio punto: no hay push
    assert comparar_casa(2.5, 44.5, TODAS)["push_spread"] == 0.0


def test_probabilidades_suman_100_con_push():
    r = comparar_casa(3.0, 44.0, TODAS)
    f = _f(r)
    assert f["sp_home"]["p_gana"] + f["sp_away"]["p_gana"] + f["sp_home"]["p_push"] == pytest.approx(100.0, abs=0.01)
    assert f["over"]["p_gana"] + f["under"]["p_gana"] + f["over"]["p_push"] == pytest.approx(100.0, abs=0.01)
    assert f["ml_home"]["p_gana"] + f["ml_away"]["p_gana"] + f["ml_home"]["p_push"] == pytest.approx(100.0, abs=0.01)


def test_casa_con_mejor_linea_da_ev_positivo_y_peor_negativo():
    # consenso: local favorito por 3. Tu casa ofrece al LOCAL -2.5 (mejor) a -110
    mejor = _f(comparar_casa(3.0, 44.0, {"sp_home": -110, "sp_away": -110}, spread_casa_local=-2.5))
    assert mejor["sp_home"]["ev"] > 0 > mejor["sp_away"]["ev"]
    # tu casa ofrece al LOCAL -3.5 (peor): al reves
    peor = _f(comparar_casa(3.0, 44.0, {"sp_home": -110, "sp_away": -110}, spread_casa_local=-3.5))
    assert peor["sp_home"]["ev"] < 0 < peor["sp_away"]["ev"]


def test_convencion_de_signos_de_la_casa():
    # local underdog en el consenso (spread_line = -2.5) y la casa lo pone +2.5 = igual linea
    r = comparar_casa(-2.5, 40.5, {"sp_home": -110, "sp_away": -110}, spread_casa_local=2.5)
    assert not r["casa_distinta_spread"]
    assert _f(r)["sp_home"]["p_gana"] == pytest.approx(50.0, abs=0.01)


def test_ml_consistente_con_el_spread():
    f = _f(comparar_casa(-2.5, 40.5, TODAS))
    # DEN +2.5: el mercado real da ~43% al local (devig del ML +120/-143 = 43.6%)
    assert 40.0 < f["ml_home"]["p_gana"] < 46.0
    assert f["ml_home"]["p_gana"] + f["ml_away"]["p_gana"] + f["ml_home"]["p_push"] == pytest.approx(100.0, abs=0.01)


def test_sin_cuotas_devuelve_none_y_solo_evalua_lo_capturado():
    assert comparar_casa(3.0, 44.0, {}) is None
    r = comparar_casa(3.0, 44.0, {"over": -110})
    assert [x["clave"] for x in r["filas"]] == ["over"]


def test_ev_y_momio_justo():
    assert ev_100(0.5, 0.0, -110) == pytest.approx(-4.55, abs=0.01)
    assert ev_100(0.5, 0.0, 100) == 0.0
    assert ev_100(0.5, 0.2, 100) == pytest.approx(-0.0 + 0.5 * 100 - 0.3 * 100, abs=0.01)  # push devuelve
    assert momio_justo(0.6) == -150 and momio_justo(0.4) == 150


def test_equilibrar_conserva_la_masa_de_empate():
    pmf = {2: 0.2, 3: 0.3, 4: 0.5}
    out = equilibrar(pmf, 3)
    assert out[3] == pytest.approx(0.3)
    assert sum(out.values()) == pytest.approx(1.0)
    assert out[2] == pytest.approx(out[4])
