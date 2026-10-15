"""find_latest_market_line nunca debe devolver la linea de un partido ya jugado."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.predict import find_latest_market_line  # noqa: E402


def _g(season, week, opp, ha):
    return {"season": season, "week": week, "opponent": opp, "home_away": ha,
            "team_score": 20.0, "opp_score": 17.0, "result": "W"}


# local A vs visitante B: S5 ya jugado, S9 sin jugar; ultimos 10 de ambos llegan a S5
LINEAS = [
    {"season": 2025, "week": 2, "home_team": "A", "away_team": "B", "spread_line": 1.0, "total_line": 40.0},
    {"season": 2026, "week": 5, "home_team": "A", "away_team": "B", "spread_line": 3.5, "total_line": 55.5},
    {"season": 2026, "week": 9, "home_team": "A", "away_team": "B", "spread_line": -2.5, "total_line": 41.0},
]
LAST10 = {
    "A": [_g(2026, 5, "B", "local")] + [_g(2026, w, "X", "local") for w in (4, 3, 2, 1)]
         + [_g(2025, w, "X", "local") for w in (18, 17, 16, 15, 14)],
    "B": [_g(2026, 5, "A", "visitante")] + [_g(2026, w, "Y", "local") for w in (4, 3, 2, 1)]
         + [_g(2025, w, "Y", "local") for w in (18, 17, 16, 15, 14)],
}


def test_ignora_linea_de_partido_ya_jugado():
    sp, tl, sw = find_latest_market_line(LINEAS, "A", "B", LAST10)
    assert sw == (2026, 9) and sp == -2.5          # S5 (jugado) y 2025 S2 (anterior a la ventana) se ignoran


def test_sin_partido_pendiente_no_devuelve_linea():
    solo_jugadas = LINEAS[:2]
    assert find_latest_market_line(solo_jugadas, "A", "B", LAST10) == (None, None, None)


def test_linea_antigua_fuera_de_la_ventana_cuenta_como_jugada():
    antigua = [LINEAS[0]]
    assert find_latest_market_line(antigua, "A", "B", LAST10) == (None, None, None)


def test_campo_played_explicito_manda():
    lin = [dict(LINEAS[2], played=True)]
    assert find_latest_market_line(lin, "A", "B", LAST10) == (None, None, None)


def test_orden_local_visitante_se_respeta():
    assert find_latest_market_line(LINEAS, "B", "A", LAST10) == (None, None, None)


def test_cache_real_bufla_jugado_y_den_sea_pendiente():
    p = ROOT / "data" / "stats_cache.json"
    if not p.exists():
        pytest.skip("sin cache")
    c = json.loads(p.read_text())
    # BUF @ LA (T2026 S5) ya se jugo: no debe haber linea
    assert find_latest_market_line(c["market_lines"], "LA", "BUF", c["last10_games"]) == (None, None, None)
    # SEA @ DEN (T2026 S6) esta pendiente
    sp, tl, sw = find_latest_market_line(c["market_lines"], "DEN", "SEA", c["last10_games"])
    assert sw == (2026, 6)
    # ninguna linea devuelta, para ningun cruce, coincide con un partido del historial
    for m in c["market_lines"]:
        res = find_latest_market_line([m], m["home_team"], m["away_team"], c["last10_games"])
        jugado = any((g["season"], g["week"], g["opponent"], g["home_away"]) ==
                     (m["season"], m["week"], m["away_team"], "local")
                     for g in c["last10_games"].get(m["home_team"], []))
        if jugado:
            assert res == (None, None, None), m
