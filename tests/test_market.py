"""Tests de coherencia del mercado y del edge.

Convencion nflverse: spread_line > 0  => el LOCAL es favorito.
Estos tests existen porque esa convencion estuvo invertida y el "mercado"
del edge salia al reves de los momios.

Correr con:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path
from statistics import mean

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.market import (  # noqa: E402
    blend_prediction,
    moneyline_to_implied_prob,
    remove_vig,
    spread_to_home_prob,
)
from api.predict import analizar_momios  # noqa: E402

# Casos reales de la evidencia: (local, visitante, spread_line nflverse, momio_local, momio_visitante)
LA_BUF = ("LA", "BUF", 3.0, -189, 150)    # LA favorito por linea y por momios
DEN_SEA = ("DEN", "SEA", -3.0, 120, -143)  # SEA favorito por linea y por momios

MODELO_FAKE = {
    "lambda_home": 25.0, "lambda_away": 22.0,
    "p_home_win": 55.0, "p_away_win": 45.0,
    "total_esperado_puntos": 47.0,
}


def _blend(spread, modelo=MODELO_FAKE, total=45.0):
    return blend_prediction(dict(modelo), spread, total)


# ---------------------------------------------------------------- (a)
def test_a_mercado_distinto_en_partidos_con_lineas_distintas():
    spreads = [-7.0, -3.5, -3.0, -2.5, 0.0, 2.5, 3.0, 3.5, 7.0]
    probs = [round(spread_to_home_prob(s) * 100, 1) for s in spreads]
    # lineas distintas -> probabilidades distintas
    assert len(set(probs)) == len(spreads)
    # y estrictamente crecientes con el spread (mas spread = local mas favorito)
    assert probs == sorted(probs)


def test_a_dos_partidos_reales_no_dan_el_mismo_mercado():
    a = _blend(LA_BUF[2])["market_home_prob"]
    b = _blend(DEN_SEA[2])["market_home_prob"]
    # |spread| igual (3.0) pero signo opuesto: deben ser espejo, NO iguales
    assert a != b
    assert a + b == pytest.approx(100.0, abs=0.2)


# ---------------------------------------------------------------- (b)
@pytest.mark.parametrize("home,away,spread,ml_home,ml_away", [LA_BUF, DEN_SEA])
def test_b_favorito_del_mercado_coincide_con_momios_y_con_la_linea(home, away, spread, ml_home, ml_away):
    pred = _blend(spread)
    fav_edge = home if pred["market_home_prob"] >= 50 else away

    dh, da = remove_vig(moneyline_to_implied_prob(ml_home), moneyline_to_implied_prob(ml_away))
    fav_momios = home if dh >= da else away
    fav_linea = home if spread > 0 else away

    assert fav_edge == fav_linea
    assert fav_edge == fav_momios


# ---------------------------------------------------------------- (c)
@pytest.mark.parametrize("home,away,spread,ml_home,ml_away", [LA_BUF, DEN_SEA])
def test_c_cada_par_suma_100(home, away, spread, ml_home, ml_away):
    pred = _blend(spread)
    assert pred["model_home_prob"] + pred["model_away_prob"] == pytest.approx(100.0, abs=0.2)
    assert pred["market_home_prob"] + pred["market_away_prob"] == pytest.approx(100.0, abs=0.2)
    assert pred["p_home_win"] + pred["p_away_win"] == pytest.approx(100.0, abs=0.2)

    mom = analizar_momios(pred, float(ml_home), float(ml_away))
    assert mom["implied_home"] + mom["implied_away"] == pytest.approx(100.0, abs=0.2)


def test_c_sin_linea_el_par_del_modelo_sigue_sumando_100():
    pred = blend_prediction(dict(MODELO_FAKE), None, None)
    assert pred["market_used"] is False
    assert pred["p_home_win"] + pred["p_away_win"] == pytest.approx(100.0, abs=0.2)


# ---------------------------------------------------------------- EV doble
def test_ev_muestra_modelo_puro_y_mezcla_por_separado():
    pred = _blend(LA_BUF[2])
    mom = analizar_momios(pred, float(LA_BUF[3]), float(LA_BUF[4]))
    for k in ("ev_home", "ev_away", "ev_home_modelo", "ev_away_modelo"):
        assert k in mom
    # el EV "modelo" usa la probabilidad pura (55/45), el otro la mezcla
    assert mom["ev_home_modelo"] != mom["ev_home"]


# ---------------------------------------------------------------- regresion del signo con datos reales
def test_signo_del_spread_contra_margenes_reales_del_cache():
    """Si alguien vuelve a invertir el signo, la correlacion implicita cambia:
    con spread_line>0 el local debe ganar por mas que con spread_line<0."""
    cache_path = ROOT / "data" / "stats_cache.json"
    if not cache_path.exists():
        pytest.skip("no hay cache local")
    cache = json.loads(cache_path.read_text())

    margen_local = {}
    for team, games in cache.get("last10_games", {}).items():
        for g in games:
            if g["home_away"] == "local":
                margen_local[(g["season"], g["week"], team, g["opponent"])] = g["team_score"] - g["opp_score"]

    pos, neg = [], []
    for m in cache.get("market_lines", []):
        key = (m["season"], m["week"], m["home_team"], m["away_team"])
        if key in margen_local and m["spread_line"]:
            (pos if m["spread_line"] > 0 else neg).append(margen_local[key])

    assert len(pos) >= 20 and len(neg) >= 20, "muy pocos partidos para validar el signo"
    assert mean(pos) > mean(neg)

    # y el codigo debe asignar mas probabilidad al local justo cuando spread_line>0
    assert spread_to_home_prob(3.0) > 0.5 > spread_to_home_prob(-3.0)
