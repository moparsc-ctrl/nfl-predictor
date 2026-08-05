"""
Job semanal: descarga datos frescos de nflverse, recalcula stats_df
y lo guarda en data/stats_cache.json para que la funcion de Vercel
solo tenga que leerlo (no recalcular en cada request).

Se ejecuta via GitHub Actions (ver .github/workflows/update-cache.yml)
"""
import json
import sys
from pathlib import Path

import nflreadpy as nfl

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.ratings import build_points_ratings, add_epa_ratings

YEARS = [2024, 2025, 2026]
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stats_cache.json"


def main():
    print("Descargando calendario...")
    schedule = nfl.load_schedules(YEARS).to_pandas()
    games = schedule.dropna(subset=["home_score", "away_score"]).copy()
    games = games[["season", "week", "home_team", "away_team", "home_score", "away_score"]]

    print("Calculando ratings por puntos...")
    stats_df, league_avg_pts = build_points_ratings(games)

    print("Descargando play-by-play para EPA (puede tardar)...")
    pbp_frames = []
    for year in YEARS:
        try:
            pbp_year = nfl.load_pbp([year]).to_pandas()
            pbp_frames.append(pbp_year)
        except Exception as e:
            print(f"  Sin play-by-play disponible todavia para {year} ({e}); se omite ese anio para EPA.")
    if pbp_frames:
        import pandas as pd
        pbp = pd.concat(pbp_frames, ignore_index=True)
        stats_df = add_epa_ratings(stats_df, pbp)
    else:
        print("  No hubo play-by-play disponible para ningun anio; se usan ratings solo por puntos.")

    print("Guardando lineas de mercado (spread/total) por partido...")
    market_cols = ["season", "week", "home_team", "away_team", "spread_line", "total_line"]
    market_df = schedule[market_cols].dropna(subset=["spread_line", "total_line"])
    market_lines = market_df.to_dict(orient="records")

    payload = {
        "league_avg_pts": float(league_avg_pts),
        "updated_years": YEARS,
        "teams": {
            team: {col: (None if pd_isna(val) else float(val)) for col, val in row.items()}
            for team, row in stats_df.to_dict(orient="index").items()
        },
        "market_lines": market_lines,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Guardado en {OUT_PATH}")


def pd_isna(val):
    try:
        import math
        return val is None or (isinstance(val, float) and math.isnan(val))
    except Exception:
        return val is None


if __name__ == "__main__":
    main()
