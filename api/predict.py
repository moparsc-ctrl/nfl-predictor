import json
import sys
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.poisson_model import predict_match, recomendacion_apuesta
from model.market import blend_prediction, moneyline_to_implied_prob, remove_vig, expected_value

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stats_cache.json"
FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

W = 1000
BG = (10, 13, 18)
PANEL = (19, 23, 31)
PANEL_BORDER = (33, 38, 48)
FG = (240, 241, 245)
MUTED = (140, 146, 158)
ACCENT = (0, 214, 143)
ACCENT_DIM = (0, 214, 143, 60)
BAR_BG = (33, 38, 48)
BAR_AWAY = (90, 130, 240)
NEG = (240, 100, 100)


ROW_H = 20


def compute_height(has_market_panel, has_odds_panel, last10_rows=0):
    y = 140
    y += 130 + 20  # forma reciente
    if last10_rows:
        y += last10_panel_height(last10_rows) + 20
    y += 130 + 20  # probabilidad
    y += 110 + 20  # marcador esperado
    y += 90 + 20   # recomendacion
    if has_market_panel:
        y += 100 + 20
    if has_odds_panel:
        y += 120 + 20
    y += 60  # espacio para el pie de pagina
    return y


def last10_panel_height(rows_n):
    header_h = 52
    avg_h = 30
    pad_bottom = 16
    return header_h + rows_n * ROW_H + avg_h + pad_bottom


def load_cache():
    payload = json.loads(DATA_PATH.read_text())
    return (
        payload["teams"],
        payload["league_avg_pts"],
        payload.get("market_lines", []),
        payload.get("last10_games", {}),
    )


def promedio_ultimos(games_list):
    """Promedio de puntos a favor/en contra sobre la lista de partidos dada
    (los ultimos N que haya disponibles, hasta 10)."""
    n = len(games_list)
    if n == 0:
        return None, None, 0
    avg_for = sum(g["team_score"] for g in games_list) / n
    avg_against = sum(g["opp_score"] for g in games_list) / n
    return round(avg_for, 1), round(avg_against, 1), n


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


def analizar_momios(pred, ml_home, ml_away):
    """Dado un par de momios americanos capturados por el usuario, calcula
    la probabilidad implicita sin margen de casa y el valor esperado (EV)
    de apostar $100 a cada lado, usando la probabilidad del modelo (ya
    mezclada con mercado si aplica) como la probabilidad 'real'."""
    implied_home = moneyline_to_implied_prob(ml_home)
    implied_away = moneyline_to_implied_prob(ml_away)
    real_home, real_away = remove_vig(implied_home, implied_away)

    ev_home = expected_value(pred["p_home_win"], ml_home)
    ev_away = expected_value(pred["p_away_win"], ml_away)

    return {
        "ml_home": ml_home,
        "ml_away": ml_away,
        "implied_home": real_home,
        "implied_away": real_away,
        "ev_home": ev_home,
        "ev_away": ev_away,
    }


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


def draw_last10_panel(d, margin, y, content_w, home, away, home_games, away_games):
    rows_n = min(10, max(len(home_games), len(away_games), 1))
    p_h = last10_panel_height(rows_n)
    panel(d, [margin, y, margin + content_w, y + p_h])
    section_label(d, margin + 24, y + 16, "Ultimos 10 partidos")

    col_w = content_w // 2
    avg_home = promedio_ultimos(home_games)
    avg_away = promedio_ultimos(away_games)

    for i, (team, rival, games_list, avg) in enumerate((
        (home, away, home_games, avg_home),
        (away, home, away_games, avg_away),
    )):
        cx = margin + 24 + i * col_w
        d.text((cx, y + 36), team, font=font(16, bold=True), fill=FG)

        row_y = y + 36 + 26
        for g in games_list[:10]:
            es_cruce_directo = g["opponent"] == rival
            vs_txt = f"vs {g['opponent']}" if g["home_away"] == "local" else f"@{g['opponent']}"
            linea = (f"T{g['season']} S{g['week']}  {vs_txt}  "
                     f"{g['result']} {g['team_score']:.0f}-{g['opp_score']:.0f}")
            color = ACCENT if es_cruce_directo else FG
            txt_font = font(13, bold=es_cruce_directo)
            if es_cruce_directo:
                # marca visual extra: pequeno indicador a la izquierda del texto
                d.text((cx, row_y), "»", font=txt_font, fill=ACCENT)
                d.text((cx + 14, row_y), linea, font=txt_font, fill=color)
            else:
                d.text((cx, row_y), linea, font=txt_font, fill=color)
            row_y += ROW_H

        avg_for, avg_against, n = avg
        if n:
            avg_txt = f"Prom. ({n}): {avg_for} a favor  ·  {avg_against} en contra"
        else:
            avg_txt = "Sin historial disponible"
        d.text((cx, y + p_h - 26), avg_txt, font=font(12, bold=True), fill=MUTED)

    return y + p_h + 20


def build_image(home, away, pred, rec, teams, season_week=None, momios=None, last10=None):
    last10 = last10 or {}
    home_games = last10.get(home, [])
    away_games = last10.get(away, [])
    last10_rows = min(10, max(len(home_games), len(away_games)))

    has_market_panel = bool(pred.get("market_used") and pred.get("edge_home") is not None)
    has_odds_panel = momios is not None
    H = compute_height(has_market_panel, has_odds_panel, last10_rows)

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

    # ---- Panel: ultimos 10 partidos ----
    if home_games or away_games:
        y = draw_last10_panel(d, margin, y, content_w, home, away, home_games, away_games)

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

    # ---- Panel: tus momios (valor real con vig removido) ----
    if momios:
        p6_h = 120
        panel(d, [margin, y, margin + content_w, y + p6_h])
        section_label(d, margin + 24, y + 16, "Tus momios (valor real)")

        linea1 = (f"Momio {home}: {momios['ml_home']:+.0f}  (implicito sin margen: {momios['implied_home']}%)   "
                  f"|   Momio {away}: {momios['ml_away']:+.0f}  (implicito sin margen: {momios['implied_away']}%)")
        d.text((margin + 24, y + 44), linea1, font=font(15), fill=FG)

        ev_home, ev_away = momios["ev_home"], momios["ev_away"]
        color_home = ACCENT if ev_home > 0 else (NEG if ev_home < 0 else MUTED)
        color_away = ACCENT if ev_away > 0 else (NEG if ev_away < 0 else MUTED)

        linea2_label = "VALOR ESPERADO APOSTANDO $100:"
        d.text((margin + 24, y + 72), linea2_label, font=font(12, bold=True), fill=MUTED)
        ev_home_txt = f"{home}: {ev_home:+.2f}"
        ev_away_txt = f"{away}: {ev_away:+.2f}"
        d.text((margin + 24, y + 92), ev_home_txt, font=font(17, bold=True), fill=color_home)
        ev_away_w = text_w(d, ev_away_txt, font(17, bold=True))
        d.text((margin + content_w - 24 - ev_away_w, y + 92), ev_away_txt, font=font(17, bold=True), fill=color_away)

        y += p6_h + 20

    d.text((margin, H - 34), "Uso informativo — no es asesoria financiera",
           font=font(12), fill=MUTED)

    return img


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        home = (qs.get("home") or [""])[0].upper()
        away = (qs.get("away") or [""])[0].upper()
        ml_home_raw = (qs.get("ml_home") or [""])[0].strip()
        ml_away_raw = (qs.get("ml_away") or [""])[0].strip()

        if not home or not away:
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Usa ?home=XXX&away=YYY (siglas de equipo, ej KC, BUF)")
            return

        teams, league_avg_pts, market_lines, last10_games = load_cache()
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

        momios = None
        if ml_home_raw and ml_away_raw:
            try:
                ml_home = float(ml_home_raw)
                ml_away = float(ml_away_raw)
                momios = analizar_momios(pred, ml_home, ml_away)
            except ValueError:
                pass  # si vienen mal formados, simplemente se omite el panel

        img = build_image(home, away, pred, rec, teams, season_week, momios, last10_games)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        self.wfile.write(buf.getvalue())
