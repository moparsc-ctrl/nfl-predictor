import json
import sys
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.poisson_model import predict_match, recomendacion_apuesta
from model.market import blend_prediction

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stats_cache.json"
FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

W, H = 1000, 900
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
    return payload["teams"], payload["league_avg_pts"], payload.get("market_lines", [])


def find_latest_market_line(market_lines, home, away):
    """Busca, entre las lineas guardadas, la mas reciente (mayor season/week)
    para ese cruce exacto (local/visitante en ese orden)."""
    candidatos = [m for m in market_lines if m["home_team"] == home and m["away_team"] == away]
    if not candidatos:
        return None, None, None
    mejor = max(candidatos, key=lambda m: (m["season"], m["week"]))
    return mejor["spread_line"], mejor["total_line"], (mejor["season"], mejor["week"])


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


def build_image(home, away, pred, rec, teams, season_week=None):
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

    if pred.get("market_used"):
        season, week = season_week
        subt = f"PROYECCION NFL  ·  MODELO POISSON + EPA  ·  AJUSTADO CON LINEA DE MERCADO (T{season} S{week})"
    else:
        subt = "PROYECCION NFL  ·  MODELO POISSON + EPA  ·  SIN LINEA DE MERCADO DISPONIBLE"
    d.text((margin, 96), subt, font=font(13, bold=True), fill=MUTED)

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

    y += p4_h + 20

    # ---- Panel: valor vs mercado (edge) ----
    if pred.get("market_used") and pred.get("edge_home") is not None:
        p5_h = 100
        panel(d, [margin, y, margin + content_w, y + p5_h])
        section_label(d, margin + 24, y + 16, "Valor vs mercado (edge)")

        edge_home, edge_away = pred["edge_home"], pred["edge_away"]
        mejor_edge = max(edge_home, edge_away)
        equipo_valor = home if edge_home >= edge_away else away
        edge_color = ACCENT if mejor_edge >= 5 else MUTED

        linea = (f"Modelo {home}: {pred['model_home_prob']}%  vs  mercado: {pred['market_home_prob']}%   "
                 f"|   Modelo {away}: {pred['model_away_prob']}%  vs  mercado: {pred['market_away_prob']}%")
        d.text((margin + 24, y + 44), linea, font=font(15), fill=FG)

        if mejor_edge >= 5:
            veredicto = f"Posible valor en {equipo_valor}: +{mejor_edge} pts vs mercado"
        else:
            veredicto = "Sin valor claro — el modelo coincide con el mercado"
        d.text((margin + 24, y + 68), veredicto, font=font(15, bold=True), fill=edge_color)

        y += p5_h + 20

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

        teams, league_avg_pts, market_lines = load_cache()
        if home not in teams or away not in teams:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Equipo no encontrado: {home if home not in teams else away}".encode())
            return

        stats_df = stats_df_like(teams)
        model_pred = predict_match(home, away, stats_df, league_avg_pts)

        spread_line, total_line, season_week = find_latest_market_line(market_lines, home, away)
        pred = blend_prediction(model_pred, spread_line, total_line)
        rec = recomendacion_apuesta(pred)

        img = build_image(home, away, pred, rec, teams, season_week)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        self.wfile.write(buf.getvalue())
