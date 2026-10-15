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
    build_image,
    parse_odds,
    parse_lineas_casa,
    preparar_ficha,
)


def main():
    if len(sys.argv) < 3:
        print("Uso: python local_test.py LOCAL VISITANTE [momio_local] [momio_visitante]")
        print("Ejemplo: python local_test.py ARI CAR +100 -121")
        sys.exit(1)

    home = sys.argv[1].upper()
    away = sys.argv[2].upper()
    # argumentos posicionales: momio_local momio_visitante; opcionales clave=valor:
    # sp_home=-110 sp_away=-110 over=-110 under=-110 (cuotas del spread y del total)
    # tambien ml_home=+120 ml_away=-143
    # sp_linea=-2.5 (linea del LOCAL en tu casa, formato casa) total_linea=41
    pos = [a for a in sys.argv[3:] if "=" not in a]
    extra = {a.split("=")[0]: [a.split("=")[1]] for a in sys.argv[3:] if "=" in a}
    ml_home_raw = pos[0] if len(pos) > 0 else (extra.get("ml_home") or [None])[0]
    ml_away_raw = pos[1] if len(pos) > 1 else (extra.get("ml_away") or [None])[0]

    teams, league_avg_pts, market_lines, last10_games = load_cache()

    if home not in teams:
        print(f"Equipo local '{home}' no encontrado en el cache. Equipos disponibles:")
        print(", ".join(sorted(teams.keys())))
        sys.exit(1)
    if away not in teams:
        print(f"Equipo visitante '{away}' no encontrado en el cache. Equipos disponibles:")
        print(", ".join(sorted(teams.keys())))
        sys.exit(1)

    ml_h = float(ml_home_raw) if ml_home_raw else None
    ml_a = float(ml_away_raw) if ml_away_raw else None
    kwargs, advertencias = preparar_ficha(home, away, (teams, league_avg_pts, market_lines, last10_games),
                                          parse_odds(extra), parse_lineas_casa(extra), ml_h, ml_a)
    for texto in advertencias:
        print(f"ADVERTENCIA: {texto}")
    if kwargs["jugado"]:
        j = kwargs["jugado"]
        print(f"Partido ya jugado (T{j['season']} S{j['week']}): {home} {j['home_score']:.0f} - "
              f"{j['away_score']:.0f} {away}. Sin recomendacion ni VE.")
    img = build_image(**kwargs)
    out_path = Path(__file__).resolve().parent / "ficha.jpg"
    img.save(out_path, quality=92)
    print(f"Listo. Ficha guardada en: {out_path}")


if __name__ == "__main__":
    main()
