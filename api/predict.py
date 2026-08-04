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
FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

W, H = 1000, 760
BG = (10, 13, 18)
PANEL = (19, 23, 31)
PANEL_BORDER = (33, 38, 48)
FG = (240, 241, 245)
MUTED = (140, 146, 158)
ACCENT = (0, 214, 143)
ACCENT_DIM = (0, 214, 143, 60)
BAR_BG = (33, 38, 48)
BAR_AWAY = (90, 130, 240)


def load_cache():
    payload = json.loads(DATA_PATH.read_text())
    return payload["teams"], payload["league_avg_pts"]


def stats_df_like(teams_dict):
    import pandas as pd
    return pd.DataFrame.from_dict(teams_dict, orient="index")


_FONT_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    f = ImageFont.truetype(str(FONTS_DIR / name), size)
    _FONT_CACHE[key] = f
    return f


def text_w(d, text, f):
    return d.textbbox((0, 0), text, font=f)[2]


def panel(d, xy, radius=16, fill=PANEL, outline=PANEL_BORDER):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1)


def section_label(d, x, y, text):
    d.text((x, y), text.upper(), font=font(14, bold=True), fill=ACCENT)


def prob_bar(d, x, y, width, height, home_pct, home_label, away_label):
    home_w = int(width * home_pct / 100)
    d.rounded_rectangle([x, y, x + width, y + height], radius=height // 2, fill=BAR_BG)
    if home_w > 2:
        d.rounded_rectangle([x, y, x + max(home_w, height), y + height], radius=height // 2, fill=ACCENT)
    if width - home_w > 2:
        d.rounded_rectangle([x + home_w, y, x + width, y + height], radius=height // 2, fill=BAR_AWAY)

    d.text((x, y + height + 8), f"{home_label}", font=font(15, bold=True), fill=ACCENT)
    away_w = text_w(d, away_label, font(15, bold=True))
    d.text((x + width - away_w, y + height + 8), f"{away_label}", font=font(15, bold=True), fill=BAR_AWAY)


def build_image(home, away, pred, rec, teams):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    margin = 44
    content_w = W - margin * 2

    # ---- Encabezado ----
    d.text((margin, 34), f"{away}", font=font(46, bold=True), fill=FG)
    away_w = text_w(d, away, font(46, bold=True))
    d.text((margin + away_w + 14, 34), "@", font=font(46), fill=MUTED)
    at_w = text_w(d, "@", font(46))
    d.text((margin + away_w + 14 + at_w + 14, 34), f"{home}", font=font(46, bold=True), fill=FG)
    d.text((margin, 96), "PROYECCION NFL  ·  MODELO POISSON + EPA", font=font(13, bold=True), fill=MUTED)

    y = 140

    # ---- Panel: forma reciente ----
    p1_h = 130
    panel(d, [margin, y, margin + content_w, y + p1_h])
    section_label(d, margin + 24, y + 18, "Forma reciente (temporada)")

    col_w = content_w // 2
    for i, team in enumerate((home, away)):
        row = teams[team]
        cx = margin + 24 + i * col_w
        role = "LOCAL" if i == 0 else "VISITANTE"
        d.text((cx, y + 48), f"{team}", font=font(24, bold=True), fill=FG)
        d.text((cx, y + 78), f"{role}", font=font(11, bold=True), fill=MUTED)
        stat_line = f"{row['pts_for_avg']:.1f} a favor  ·  {row['pts_against_avg']:.1f} en contra"
        d.text((cx, y + 96), stat_line, font=font(15), fill=FG)

    y += p1_h + 20

    # ---- Panel: probabilidad ----
    p2_h = 130
    panel(d, [margin, y, margin + content_w, y + p2_h])
    section_label(d, margin + 24, y + 18, "Probabilidad de resultado")
    prob_bar(
        d, margin + 24, y + 50, content_w - 48, 22,
        pred["p_home_win"],
        f"{home} {pred['p_home_win']}%",
        f"{pred['p_away_win']}% {away}",
    )

    y += p2_h + 20

    # ---- Panel: marcador esperado + total ----
    p3_h = 110
    panel(d, [margin, y, margin + content_w, y + p3_h])
    section_label(d, margin + 24, y + 18, "Marcador esperado")
    marcador = f"{home}  {pred['lambda_home']}   -   {pred['lambda_away']}  {away}"
    d.text((margin + 24, y + 46), marcador, font=font(30, bold=True), fill=FG)

    total_label = "TOTAL COMBINADO (OVER/UNDER)"
    total_val = f"{pred['total_esperado_puntos']} pts"
    tw = text_w(d, total_label, font(12, bold=True))
    vw = text_w(d, total_val, font(22, bold=True))
    right_x = margin + content_w - 24
    d.text((right_x - tw, y + 46), total_label, font=font(12, bold=True), fill=MUTED)
    d.text((right_x - vw, y + 66), total_val, font=font(22, bold=True), fill=FG)

    y += p3_h + 20

    # ---- Panel: recomendacion ----
    p4_h = 90
    panel(d, [margin, y, margin + content_w, y + p4_h], fill=(13, 30, 25), outline=(0, 90, 60))
    section_label(d, margin + 24, y + 16, "Recomendacion")
    d.text((margin + 24, y + 42), rec, font=font(22, bold=True), fill=ACCENT)

    d.text((margin, H - 34), "Uso informativo — no es asesoria financiera",
           font=font(12), fill=MUTED)

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
        img.save(buf, format="JPEG", quality=92)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        self.wfile.write(buf.getvalue())
