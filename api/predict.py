"""
GET /api/predict?home=KC&away=BUF

Devuelve una ficha JPG con: forma reciente, probabilidades (Poisson+EPA),
total esperado de puntos (para over/under) y recomendacion de apuesta.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.poisson_model import predict_match, recomendacion_apuesta

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stats_cache.json"

W, H = 900, 700
BG = (14, 17, 23)
FG = (240, 240, 240)
ACCENT = (0, 200, 120)
MUTED = (150, 155, 165)


def load_cache():
    payload = json.loads(DATA_PATH.read_text())
    return payload["teams"], payload["league_avg_pts"]


def stats_df_like(teams_dict):
    # pequeno wrapper para que predict_match pueda usar .loc[team, col]
    import pandas as pd
    return pd.DataFrame.from_dict(teams_dict, orient="index")


def font(size, bold=False):
    try:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold \
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def build_image(home, away, pred, rec, teams):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((40, 30), f"{away} @ {home}", font=font(38, bold=True), fill=FG)
    d.text((40, 80), "Proyeccion NFL — modelo Poisson + EPA", font=font(16), fill=MUTED)

    # Forma reciente (puntos por partido, ya calculado en el cache)
    y = 140
    d.text((40, y), "Forma reciente (temporada)", font=font(20, bold=True), fill=ACCENT)
    y += 36
    for team in (home, away):
        row = teams[team]
        line = (f"{team}:  {row['pts_for_avg']:.1f} pts/partido a favor  |  "
                f"{row['pts_against_avg']:.1f} en contra  ({int(row['n_partidos'])} PJ)")
        d.text((40, y), line, font=font(18), fill=FG)
        y += 30

    # Probabilidades
    y += 20
    d.text((40, y), "Probabilidad de resultado (Poisson)", font=font(20, bold=True), fill=ACCENT)
    y += 36
    d.text((40, y), f"{home} gana: {pred['p_home_win']}%", font=font(20), fill=FG)
    y += 28
    d.text((40, y), f"{away} gana: {pred['p_away_win']}%", font=font(20), fill=FG)

    # Marcador esperado / total
    y += 50
    d.text((40, y), "Marcador esperado", font=font(20, bold=True), fill=ACCENT)
    y += 36
    d.text((40, y), f"{home} {pred['lambda_home']}  -  {pred['lambda_away']} {away}",
           font=font(22), fill=FG)
    y += 34
    d.text((40, y), f"Total combinado esperado (over/under): {pred['total_esperado_puntos']} pts",
           font=font(18), fill=FG)

    # Recomendacion
    y += 60
    d.text((40, y), "Recomendacion", font=font(20, bold=True), fill=ACCENT)
    y += 36
    d.text((40, y), rec, font=font(20), fill=FG)

    d.text((40, H - 40), "Uso informativo — no es asesoria financiera",
           font=font(13), fill=MUTED)
    return img


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        home = (qs.get("home") or [""])[0].upper()
        away = (qs.get("away") or [""])[0].upper()

        if not home or not away:
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Usa ?home=XXX&away=YYY (siglas de equipo, ej KC, BUF)")
            return

        teams, league_avg_pts = load_cache()
        if home not in teams or away not in teams:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Equipo no encontrado: {home if home not in teams else away}".encode())
            return

        stats_df = stats_df_like(teams)
        pred = predict_match(home, away, stats_df, league_avg_pts)
        rec = recomendacion_apuesta(pred)

        img = build_image(home, away, pred, rec, teams)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        self.wfile.write(buf.getvalue())
