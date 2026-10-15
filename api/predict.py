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
EDGE_H = 136
ODDS_H = 166


def compute_height(has_market_panel, has_odds_panel, last10_rows=0):
    y = 140
    y += 130 + 20  # forma reciente
    if last10_rows:
        y += last10_panel_height(last10_rows) + 20
    y += 130 + 20  # probabilidad
    y += 110 + 20  # marcador esperado
    y += 90 + 20   # recomendacion
    if has_market_panel:
        y += EDGE_H + 20
    if has_odds_panel:
        y += ODDS_H + 20
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
    de apostar $100 a cada lado, con DOS probabilidades 'reales':

    - modelo puro (independiente del mercado): ev_*_modelo
    - mezcla 65% mercado + 35% modelo (la de la barra): ev_*

    La mezcla ya contiene al mercado, asi que su EV tiende a ser circular;
    por eso se muestran ambos, etiquetados.
    """
    implied_home = moneyline_to_implied_prob(ml_home)
    implied_away = moneyline_to_implied_prob(ml_away)
    real_home, real_away = remove_vig(implied_home, implied_away)

    ev_home = expected_value(pred["p_home_win"], ml_home)
    ev_away = expected_value(pred["p_away_win"], ml_away)

    # sin linea de mercado, pred ya ES el modelo puro
    p_home_modelo = pred.get("model_home_prob", pred["p_home_win"])
    p_away_modelo = pred.get("model_away_prob", pred["p_away_win"])
    ev_home_modelo = expected_value(p_home_modelo, ml_home)
    ev_away_modelo = expected_value(p_away_modelo, ml_away)

    return {
        "ml_home": ml_home,
        "ml_away": ml_away,
        "implied_home": real_home,
        "implied_away": real_away,
        "ev_home": ev_home,
        "ev_away": ev_away,
        "ev_home_modelo": ev_home_modelo,
        "ev_away_modelo": ev_away_modelo,
        "has_blend": bool(pred.get("market_used")),
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
    if pred.get("market_used"):
        titulo_prob = "Probabilidad de resultado · MEZCLA (65% mercado + 35% modelo)"
    else:
        titulo_prob = "Probabilidad de resultado · MODELO PURO (sin linea de mercado)"
    section_label(d, margin + 24, y + 18, titulo_prob)
    prob_bar(
        d, margin + 24, y + 50, content_w - 48, 22,
        pred["p_home_win"],
        f"{home} {pred['p_home_win']}%",
        f"{pred['p_away_win']}% {away}",
    )
    if pred.get("market_used"):
        ref = (f"Modelo puro: {home} {pred['model_home_prob']}% / {away} {pred['model_away_prob']}%   ·   "
               f"Mercado (spread): {home} {pred['market_home_prob']}% / {away} {pred['market_away_prob']}%")
        d.text((margin + 24, y + 104), ref, font=font(12), fill=MUTED)

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
        p5_h = EDGE_H
        panel(d, [margin, y, margin + content_w, y + p5_h])
        section_label(d, margin + 24, y + 16, "Valor vs mercado (edge)")

        edge_home, edge_away = pred["edge_home"], pred["edge_away"]
        mejor_edge = max(edge_home, edge_away)
        equipo_valor = home if edge_home >= edge_away else away
        edge_color = ACCENT if mejor_edge >= 5 else MUTED

        # tabla: equipo | MODELO PURO | MERCADO (spread) | DIFERENCIA (modelo - mercado)
        x_left = margin + 24
        cols = (margin + 24 + 400, margin + 24 + 620, margin + content_w - 24)
        for cx, head in zip(cols, ("MODELO PURO", "MERCADO (SPREAD)", "DIFERENCIA")):
            d.text((cx - text_w(d, head, font(11, bold=True)), y + 40), head, font=font(11, bold=True), fill=MUTED)

        filas = (
            (home, pred["model_home_prob"], pred["market_home_prob"], edge_home),
            (away, pred["model_away_prob"], pred["market_away_prob"], edge_away),
        )
        for i, (team, p_mod, p_mkt, diff) in enumerate(filas):
            ry = y + 60 + i * 24
            d.text((x_left, ry), team, font=font(16, bold=True), fill=FG)
            for cx, txt, color in (
                (cols[0], f"{p_mod}%", FG),
                (cols[1], f"{p_mkt}%", FG),
                (cols[2], f"{diff:+.1f} pts",
                 ACCENT if diff >= 5 else (NEG if diff <= -5 else MUTED)),
            ):
                d.text((cx - text_w(d, txt, font(16, bold=True)), ry), txt, font=font(16, bold=True), fill=color)

        if mejor_edge >= 5:
            veredicto = f"Posible valor en {equipo_valor}: +{mejor_edge} pts vs mercado"
        else:
            veredicto = "Sin valor claro — el modelo coincide con el mercado"
        d.text((x_left, y + 112), veredicto, font=font(15, bold=True), fill=edge_color)

        y += p5_h + 20

    # ---- Panel: tus momios (valor real con vig removido) ----
    if momios:
        p6_h = ODDS_H
        panel(d, [margin, y, margin + content_w, y + p6_h])
        section_label(d, margin + 24, y + 16, "Tus momios (valor real)")

        linea1 = (f"Momio {home}: {momios['ml_home']:+.0f}  (implicito sin margen: {momios['implied_home']}%)   "
                  f"|   Momio {away}: {momios['ml_away']:+.0f}  (implicito sin margen: {momios['implied_away']}%)")
        d.text((margin + 24, y + 40), linea1, font=font(15), fill=FG)

        x_left = margin + 24
        right = margin + content_w - 24
        d.text((x_left, y + 74), "VALOR ESPERADO APOSTANDO $100", font=font(11, bold=True), fill=MUTED)

        # columnas de EV: (borde derecho, encabezado, clave home, clave away)
        if momios.get("has_blend"):
            columnas = (
                (right - 250, "CON MODELO PURO", "ev_home_modelo", "ev_away_modelo"),
                (right, "CON MEZCLA 65/35 (BARRA)", "ev_home", "ev_away"),
            )
        else:
            columnas = ((right, "CON MODELO PURO (SIN LINEA DE MERCADO)", "ev_home_modelo", "ev_away_modelo"),)

        for cx, head, _, _ in columnas:
            d.text((cx - text_w(d, head, font(11, bold=True)), y + 74), head, font=font(11, bold=True), fill=MUTED)

        for i, (team, ml, k_idx) in enumerate(((home, momios["ml_home"], 2), (away, momios["ml_away"], 3))):
            ry = y + 94 + i * 24
            d.text((x_left, ry), f"{team} ({ml:+.0f})", font=font(16, bold=True), fill=FG)
            for col in columnas:
                cx, key = col[0], col[k_idx]
                val = momios[key]
                color = ACCENT if val > 0 else (NEG if val < 0 else MUTED)
                txt = f"{val:+.2f}"
                d.text((cx - text_w(d, txt, font(17, bold=True)), ry), txt, font=font(17, bold=True), fill=color)

        if momios.get("has_blend"):
            d.text((x_left, y + 146),
                   "La mezcla ya incluye al mercado: su EV es circular. El EV del modelo puro es la señal independiente.",
                   font=font(11), fill=MUTED)

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
