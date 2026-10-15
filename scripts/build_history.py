"""Genera data/history.json: partidos de temporada regular 2015+ con la linea de
cierre (spread_line, total_line) y el resultado (margen del local, total de puntos).

Lo usa model/lineas.py para estimar, con frecuencias historicas, la probabilidad de
cada margen (3, 7, empate exacto contra el spread, etc.) condicionada a la linea.
Convencion nflverse: spread_line > 0 = el local es favorito.

Uso:  python scripts/build_history.py [anio_inicial] [anio_final]
"""
import json
import sys
from pathlib import Path

import nflreadpy as nfl

OUT = Path(__file__).resolve().parent.parent / "data" / "history.json"


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    s = nfl.load_schedules(list(range(y0, y1 + 1))).to_pandas()
    s = s[(s.game_type == "REG")].dropna(subset=["home_score", "away_score", "spread_line", "total_line"])
    rows = [[int(r.season), int(r.week), float(r.spread_line), float(r.total_line),
             int(r.home_score - r.away_score), int(r.home_score + r.away_score)]
            for r in s.itertuples()]
    OUT.write_text(json.dumps({
        "columns": ["season", "week", "spread_line", "total_line", "margin", "total"],
        "fuente": f"nflverse {y0}-{y1}, temporada regular",
        "rows": rows,
    }, separators=(",", ":")))
    print(f"{len(rows)} partidos -> {OUT}")


if __name__ == "__main__":
    main()
