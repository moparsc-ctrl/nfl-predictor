"""Advertencias de datos de entrada, cuotas por defecto y partido ya jugado."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.predict import (  # noqa: E402
    build_image, compute_height, load_cache, preparar_ficha, resolver_partido,
)
from model.lineas import TEXTO_VE_ALTO, comparar_casa  # noqa: E402

BASE = {"sp_home": -110, "sp_away": -110, "over": -110, "under": -110}


def _tipos(r):
    return [a["tipo"] for a in r["advertencias"]]


# ------------------------------------------------------------ (2) linea invertida
def test_linea_invertida_consenso_den_mas_2_5_contra_casa_den_menos_3():
    # consenso DEN +2.5 (spread_line nflverse = -2.5); tu casa DEN -3.0 -> signo opuesto, difiere 5.5
    r = comparar_casa(-2.5, 40.5, BASE, spread_casa_local=-3.0, equipo_local="DEN")
    assert "linea_invertida" in _tipos(r)
    texto = [a["texto"] for a in r["advertencias"] if a["tipo"] == "linea_invertida"][0]
    assert "invertida" in texto and "+2.5" in texto and "-3.0" in texto and "DEN" in texto


def test_misma_linea_sin_inversion_no_avisa():
    # el caso real del usuario: DEN +3 en su casa contra DEN +2.5 en el consenso
    assert "linea_invertida" not in _tipos(comparar_casa(-2.5, 40.5, BASE, spread_casa_local=3.0))
    assert "linea_invertida" not in _tipos(comparar_casa(-2.5, 40.5, BASE))  # sin linea capturada


def test_signo_opuesto_pero_diferencia_pequena_no_avisa():
    # consenso DEN +1, casa DEN -1: signo opuesto pero solo 2 pts de diferencia
    assert "linea_invertida" not in _tipos(comparar_casa(-1.0, 40.5, BASE, spread_casa_local=-1.0))


def test_mismo_signo_con_gran_diferencia_no_es_inversion():
    assert "linea_invertida" not in _tipos(comparar_casa(2.5, 40.5, BASE, spread_casa_local=-7.0))  # cons -2.5 vs -7: mismo signo


# ------------------------------------------------------------ (3) VE inusual
def test_ve_mayor_a_10_dispara_advertencia():
    r = comparar_casa(-2.5, 40.5, {"ml_home": 300, "ml_away": -400})  # DEN +300 con ~43%: VE enorme
    assert max(f["ev"] for f in r["filas"]) > 10
    assert TEXTO_VE_ALTO in [a["texto"] for a in r["advertencias"]]
    assert TEXTO_VE_ALTO == "VE inusualmente alto, revisa los datos de entrada"


def test_ve_normal_no_avisa():
    r = comparar_casa(-2.5, 40.5, BASE)
    assert max(f["ev"] for f in r["filas"]) <= 10
    assert "ve_alto" not in _tipos(r)


# ------------------------------------------------------------ (4) cuotas capturadas vs por defecto
def test_cuotas_por_defecto_se_marcan():
    r = comparar_casa(-2.5, 40.5, {"sp_home": -122, "over": -106})
    f = {x["clave"]: x for x in r["filas"]}
    assert not f["sp_home"]["cuota_estandar"] and not f["over"]["cuota_estandar"]
    assert f["sp_away"]["cuota_estandar"] and f["sp_away"]["cuota"] == -110
    assert f["under"]["cuota_estandar"]
    # lo que capturaste completo no lleva asterisco
    assert not any(x["cuota_estandar"] for x in comparar_casa(-2.5, 40.5, BASE)["filas"])


# ------------------------------------------------------------ (5) partido ya jugado
def _cache():
    return load_cache()


def test_resolver_partido_jugado_y_pendiente_con_cache_real():
    teams, lg, ml, l10 = _cache()
    j = resolver_partido(ml, "LA", "BUF", l10)
    assert j["estado"] == "jugado" and (j["home_score"], j["away_score"]) == (34.0, 37.0)
    assert (j["season"], j["week"]) == (2026, 5)
    p = resolver_partido(ml, "DEN", "SEA", l10)
    assert p["estado"] == "pendiente" and p["season_week"] == (2026, 6)
    assert resolver_partido(ml, "KC", "NYG", l10)["estado"] in ("sin_linea", "pendiente")


def test_partido_jugado_no_genera_recomendacion_ni_ve():
    cache = _cache()
    odds = {"sp_home": -110, "sp_away": -110, "over": -110, "under": -110}
    kw, adv = preparar_ficha("LA", "BUF", cache, odds, {"sp_linea": -3.5}, -189.0, 150.0)
    assert kw["jugado"]["home_score"] == 34.0
    assert kw["rec"] is None and kw["momios"] is None
    assert kw["cobertura"] is None and kw["comparacion"] is None and adv == []
    img = build_image(**kw)  # no debe fallar sin recomendacion
    assert img.size[1] < compute_height(True, True, 10, 2, True, True, 6)
    # el partido pendiente si trae todo
    kw2, _ = preparar_ficha("DEN", "SEA", cache, odds, {}, 120.0, -143.0)
    assert kw2["jugado"] is None and kw2["rec"] is not None
    assert kw2["comparacion"] is not None and kw2["momios"] is not None


def test_ficha_con_advertencias_dibuja_y_las_devuelve():
    cache = _cache()
    kw, adv = preparar_ficha("DEN", "SEA", cache, dict(BASE), {"sp_linea": -3.0}, None, None)
    assert any("invertida" in a for a in adv)
    build_image(**kw)


def test_consola_local_test_muestra_advertencia_y_partido_jugado():
    py = [sys.executable, "-I", str(ROOT / "local_test.py")]
    out = subprocess.run(py + ["DEN", "SEA", "sp_linea=-3", "sp_home=-110", "sp_away=-110"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "ADVERTENCIA" in out.stdout and "invertida" in out.stdout
    out = subprocess.run(py + ["LA", "BUF"], capture_output=True, text=True, cwd=ROOT)
    assert "Partido ya jugado" in out.stdout and "34 - 37" in out.stdout
