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
    analizar_spread_total,
    parse_odds,
    parse_lineas_casa,
    construir_comparacion,
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
    # argumentos posicionales: momio_local momio_visitante; opcionales clave=valor:
    # sp_home=-110 sp_away=-110 over=-110 under=-110 (cuotas del spread y del total)
    # sp_linea=-2.5 (linea del LOCAL en tu casa, formato casa) total_linea=41
    pos = [a for a in sys.argv[3:] if "=" not in a]
    extra = {a.split("=")[0]: [a.split("=")[1]] for a in sys.argv[3:] if "=" in a}
    ml_home_raw = pos[0] if len(pos) > 0 else None
    ml_away_raw = pos[1] if len(pos) > 1 else None

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

    spread_line, total_line, season_week = find_latest_market_line(market_lines, home, away, last10_games)
    pred = blend_prediction(model_pred, spread_line, total_line)
    rec = recomendacion_apuesta(pred, home, away)
    cobertura = analizar_spread_total(pred, parse_odds(extra))

    momios = None
    if ml_home_raw and ml_away_raw:
        try:
            momios = analizar_momios(pred, float(ml_home_raw), float(ml_away_raw))
        except ValueError:
            print("Momios invalidos, se omite ese panel.")

    comparacion = construir_comparacion(spread_line, total_line, parse_odds(extra), momios,
                                        parse_lineas_casa(extra))
    img = build_image(home, away, pred, rec, teams, season_week, momios, last10_games, cobertura,
                      comparacion)
    out_path = Path(__file__).resolve().parent / "ficha.jpg"
    img.save(out_path, quality=92)
    print(f"Listo. Ficha guardada en: {out_path}")


if __name__ == "__main__":
    main()
