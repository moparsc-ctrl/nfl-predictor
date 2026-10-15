"""
Prueba local de la ficha SIN pasar por Vercel (evita el bug de
'vercel dev' con rutas en Windows).

Uso:
    python local_test.py ARI CAR
    python local_test.py ARI CAR +100 -121

El primer equipo es el LOCAL, el segundo el VISITANTE.
Los momios (opcionales) van en formato americano.

Genera un archivo ficha.jpg en esta misma carpeta que puedes abrir
con doble clic.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from api.predict import (
    load_cache,
    stats_df_like,
    find_latest_market_line,
    analizar_momios,
    build_image,
)
from model.poisson_model import predict_match, recomendacion_apuesta
from model.market import blend_prediction


def main():
    if len(sys.argv) < 3:
        print("Uso: python local_test.py LOCAL VISITANTE [momio_local] [momio_visitante]")
        print("Ejemplo: python local_test.py ARI CAR +100 -121")
        sys.exit(1)

    home = sys.argv[1].upper()
    away = sys.argv[2].upper()
    ml_home_raw = sys.argv[3] if len(sys.argv) > 3 else None
    ml_away_raw = sys.argv[4] if len(sys.argv) > 4 else None

    teams, league_avg_pts, market_lines, last10_games = load_cache()

    if home not in teams:
        print(f"Equipo local '{home}' no encontrado en el cache. Equipos disponibles:")
        print(", ".join(sorted(teams.keys())))
        sys.exit(1)
    if away not in teams:
        print(f"Equipo visitante '{away}' no encontrado en el cache. Equipos disponibles:")
        print(", ".join(sorted(teams.keys())))
        sys.exit(1)

    stats_df = stats_df_like(teams)
    model_pred = predict_match(home, away, stats_df, league_avg_pts)

    spread_line, total_line, season_week = find_latest_market_line(market_lines, home, away)
    pred = blend_prediction(model_pred, spread_line, total_line)
    rec = recomendacion_apuesta(pred, home, away)

    momios = None
    if ml_home_raw and ml_away_raw:
        try:
            momios = analizar_momios(pred, float(ml_home_raw), float(ml_away_raw))
        except ValueError:
            print("Momios invalidos, se omite ese panel.")

    img = build_image(home, away, pred, rec, teams, season_week, momios, last10_games)
    out_path = Path(__file__).resolve().parent / "ficha.jpg"
    img.save(out_path, quality=92)
    print(f"Listo. Ficha guardada en: {out_path}")


if __name__ == "__main__":
    main()
