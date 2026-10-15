"""Margen negativo, VE alto en 'Tus momios' y panel spread/total sin mezclar fuentes."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.predict import analizar_momios, build_image, load_cache, preparar_ficha  # noqa: E402
from model.lineas import TEXTO_VE_ALTO, comparar_casa, margen_de_casa  # noqa: E402
from model.market import analizar_spread_total, blend_prediction  # noqa: E402

MODELO = {"lambda_home": 21.0, "lambda_away": 20.0, "p_home_win": 52.0, "p_away_win": 48.0,
          "total_esperado_puntos": 41.0, "margen_esperado": 1.0, "metodo": "Normal σ=13.6",
          "sigma_margen": 13.6, "sigma_total": 13.9}
# consenso DEN +2.5 (spread_line nflverse -2.5), total 40.5
PRED = blend_prediction(dict(MODELO), -2.5, 40.5)
OK = {"sp_home": -110, "sp_away": -110, "over": -110, "under": -110, "ml_home": 120, "ml_away": -143}


def _textos(r):
    return [a["texto"] for a in r["advertencias"]]


# ------------------------------------------------------------------ 1. margen negativo
def test_margen_de_casa():
    assert margen_de_casa(-110, -110) == pytest.approx(4.8, abs=0.05)
    assert margen_de_casa(300, -143) == pytest.approx(-16.2, abs=0.05)  # el caso de la ficha de prueba


@pytest.mark.parametrize("mercado,cuotas", [
    ("spread", {"sp_home": 105, "sp_away": 105}),      # +105/+105 -> margen negativo
    ("total", {"over": 110, "under": 110}),
    ("ganador", {"ml_home": 300, "ml_away": -143}),
])
def test_margen_negativo_dispara_advertencia_en_cada_mercado(mercado, cuotas):
    r = comparar_casa(-2.5, 40.5, cuotas)
    assert r["margen_casa"][mercado] < 0
    assert f"Margen negativo en {mercado}: revisa las cuotas capturadas" in _textos(r)


def test_margen_normal_no_avisa_y_cuota_por_defecto_no_cuenta_como_margen():
    assert not any("Margen negativo" in t for t in _textos(comparar_casa(-2.5, 40.5, OK)))
    # una sola cuota capturada (la otra va por defecto): no hay margen de tu casa que evaluar
    r = comparar_casa(-2.5, 40.5, {"sp_home": 120})
    assert r["margen_casa"]["spread"] is None


def test_margen_negativo_llega_a_la_consola_y_a_la_ficha():
    kw, adv = preparar_ficha("DEN", "SEA", load_cache(), {}, {}, 300.0, -143.0)
    assert "Margen negativo en ganador: revisa las cuotas capturadas" in adv
    build_image(**kw)
    out = subprocess.run([sys.executable, "-I", str(ROOT / "local_test.py"), "DEN", "SEA",
                          "ml_home=+300", "ml_away=-143"], capture_output=True, text=True, cwd=ROOT)
    assert "Margen negativo en ganador" in out.stdout


# ------------------------------------------------------------------ 2. VE alto en "Tus momios"
def test_ve_alto_en_tus_momios_modelo_puro_y_barra():
    # sin mezcla de mercado, el modelo puro da 52% al local: +300 -> EV enorme
    m = analizar_momios(blend_prediction(dict(MODELO), -2.5, 40.5, market_weight=0.65), 300.0, -143.0)
    assert m["ev_home_modelo"] > 10 and m["ev_home"] > 10
    assert TEXTO_VE_ALTO in m["advertencias"]


def test_ve_alto_solo_en_modelo_puro_tambien_avisa():
    # con peso 100% la barra es el mercado (EV chico) pero el modelo puro sigue alto
    m = analizar_momios(blend_prediction(dict(MODELO), -2.5, 40.5, market_weight=1.0), 120.0, -143.0)
    assert m["ev_home_modelo"] > 10 and m["ev_home"] <= 10
    assert TEXTO_VE_ALTO in m["advertencias"]


def test_momios_normales_no_avisan():
    cerca = dict(MODELO, p_home_win=44.0, p_away_win=56.0, margen_esperado=-1.0)   # modelo cerca del mercado
    m = analizar_momios(blend_prediction(cerca, -2.5, 40.5), 120.0, -143.0)
    assert max(m["ev_home"], m["ev_away"], m["ev_home_modelo"], m["ev_away_modelo"]) <= 10
    assert m["advertencias"] == []


def test_ve_alto_de_momios_llega_a_consola():
    out = subprocess.run([sys.executable, "-I", str(ROOT / "local_test.py"), "DEN", "SEA",
                          "ml_home=+300", "ml_away=-143"], capture_output=True, text=True, cwd=ROOT)
    assert TEXTO_VE_ALTO in out.stdout


# ------------------------------------------------------------------ 3. spread/total sin mezclar fuentes
CASA = {"sp_home": -122, "sp_away": 100, "over": -106, "under": -115}
LINEAS = {"sp_linea": 3.0, "total_linea": 41.0}


def _f(r):
    return {x["clave"]: x for x in r["filas"]}


def test_con_datos_de_casa_usa_mi_linea_y_mis_cuotas():
    r = analizar_spread_total(PRED, CASA, LINEAS)
    f = _f(r)
    assert r["usa_casa"] and not r["mixto"]
    assert (f["sp_home"]["linea"], f["sp_away"]["linea"]) == (3.0, -3.0)      # DEN +3 / SEA -3
    assert f["over"]["linea"] == f["under"]["linea"] == 41.0
    assert [f[k]["cuota"] for k in ("sp_home", "sp_away", "over", "under")] == [-122, 100, -106, -115]
    assert not any(x["cuota_estandar"] for x in r["filas"])
    assert all(x["fuente"] == "casa" for x in r["filas"])
    assert f["sp_home"]["p_push"] > 5   # SEA -3 / DEN +3: el 3 exacto es un push real


def test_sin_datos_usa_consenso_con_menos_110_marcado():
    r = analizar_spread_total(PRED, {}, {})
    f = _f(r)
    assert not r["usa_casa"]
    assert (f["sp_home"]["linea"], f["over"]["linea"]) == (2.5, 40.5)          # consenso
    assert all(x["cuota"] == -110 and x["cuota_estandar"] and x["fuente"] == "consenso" for x in r["filas"])


def test_nunca_mezcla_cuotas_de_casa_con_linea_de_consenso():
    # capturaste cuotas pero NO tus lineas: esas cuotas no se aplican a 2.5 / 40.5
    r = analizar_spread_total(PRED, CASA, {})
    for x in r["filas"]:
        assert x["fuente"] == "consenso" and x["cuota"] == -110 and x["cuota_estandar"]
    assert any("sin sp_linea" in n for n in r["notas"]) and any("sin total_linea" in n for n in r["notas"])


def test_nunca_mezcla_linea_de_casa_con_cuotas_inventadas_del_consenso():
    # solo capturaste la linea del spread: ese mercado es de tu casa (cuotas por defecto
    # marcadas) y el total sigue siendo del consenso; cada fila lleva una sola fuente
    r = analizar_spread_total(PRED, {"sp_home": -122, "over": -106}, {"sp_linea": 3.0})
    f = _f(r)
    assert f["sp_home"]["fuente"] == "casa" and f["sp_home"]["cuota"] == -122 and not f["sp_home"]["cuota_estandar"]
    assert f["sp_away"]["fuente"] == "casa" and f["sp_away"]["cuota_estandar"]
    assert f["over"]["fuente"] == "consenso" and f["over"]["cuota"] == -110 and f["over"]["cuota_estandar"]
    assert f["over"]["linea"] == 40.5 and r["mixto"]
    for x in r["filas"]:   # invariante: fuente casa => linea tuya; fuente consenso => cuota por defecto
        if x["fuente"] == "consenso":
            assert x["cuota_estandar"]


def test_tu_linea_cambia_el_resultado_respecto_al_consenso():
    casa = _f(analizar_spread_total(PRED, CASA, LINEAS))
    cons = _f(analizar_spread_total(PRED, {}, {}))
    # con margenes enteros DEN +3 y DEN +2.5 cubren igual, pero con +3 el 3 exacto es push
    assert casa["sp_home"]["p_push"] > 5 and cons["sp_home"]["p_push"] == 0
    # SEA -3 gana menos seguido que SEA -2.5
    assert casa["sp_away"]["prob"] < cons["sp_away"]["prob"]


def test_titulos_y_dibujo_con_y_sin_datos_de_casa():
    cache = load_cache()
    kw, _ = preparar_ficha("DEN", "SEA", cache, dict(CASA), dict(LINEAS), 120.0, -143.0)
    assert kw["cobertura"]["usa_casa"]
    build_image(**kw)
    kw, _ = preparar_ficha("DEN", "SEA", cache, {}, {}, None, None)
    assert not kw["cobertura"]["usa_casa"]
    build_image(**kw)


# ------------------------------------------------------------------ subtitulo refleja la config real
def test_subtitulo_refleja_usar_epa():
    import pandas as pd
    from api.predict import subtitulo
    from model import config as cfg
    from model.poisson_model import predict_match
    row = {"off_strength": 1.0, "def_strength": 1.0, "off_strength_epa": 1.1, "def_strength_epa": 0.9}
    df = pd.DataFrame({"AAA": row, "BBB": row}).T
    sin = blend_prediction(predict_match("AAA", "BBB", df, 22.0, use_epa=False), -3.0, 44.0)
    con = blend_prediction(predict_match("AAA", "BBB", df, 22.0, use_epa=True), -3.0, 44.0)
    assert "SIN EPA" in subtitulo(sin, (2026, 6)) and "+ EPA" not in subtitulo(sin, (2026, 6))
    assert "POISSON + EPA" in subtitulo(con, (2026, 6))
    # por defecto sigue config.USAR_EPA (False): la ficha real no debe decir EPA
    por_defecto = blend_prediction(predict_match("AAA", "BBB", df, 22.0), -3.0, 44.0)
    assert ("POISSON + EPA" in subtitulo(por_defecto, (2026, 6))) == cfg.USAR_EPA
    kw, _ = preparar_ficha("DEN", "SEA", load_cache(), {}, {}, None, None)
    assert ("EPA" in subtitulo(kw["pred"], kw["season_week"]).replace("SIN EPA", "")) == cfg.USAR_EPA
