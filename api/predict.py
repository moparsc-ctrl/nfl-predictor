import json
import sys
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.poisson_model import predict_match, recomendacion_apuesta, UMBRAL_VALOR
from model.lineas import (comparar_casa, margen_de_casa, texto_margen_negativo,
                          TEXTO_VE_ALTO, VE_INUSUAL)
from model.market import (blend_prediction, marcadores, mezcla_texto, analizar_spread_total,
                          moneyline_to_implied_prob, remove_vig, expected_value)

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
WARN = (235, 185, 70)


ROW_H = 20
EDGE_H = 136
ODDS_H = 166
MARC_ROW_H = 58
REC_H = 112
REC_ALERT_EXTRA = 26
COVER_BASE_H = 180
COVER_NOTE_H = 16
ODDS_ADV_H = 22


def cobertura_notas(cobertura):
    """Lineas de nota al pie del panel de spread/total (todas dependen de las fuentes usadas)."""
    notas = []
    if any(f["cuota_estandar"] for f in cobertura["filas"]):
        notas.append("* cuota por defecto -110 (no capturada); sin asterisco = la capturaste tu.")
    if cobertura["usa_casa"]:
        notas.append("Tus lineas y tus cuotas; empate en numeros clave segun factores historicos 2015-2020.")
    if not cobertura["usa_casa"] or cobertura["mixto"]:
        notas.append("Filas del CONSENSO usan cuotas -110: no se mezclan con cuotas de tu casa.")
    notas += cobertura["notas"]
    notas.append("En backtest 2018-2025 apostar por estas diferencias no dio ROI positivo.")
    return notas


def cobertura_height(n_notas):
    return COVER_BASE_H + COVER_NOTE_H * n_notas
COMP_BASE_H = 150
COMP_ROW_H = 24


PLAYED_H = 100


def comparacion_height(n_filas, n_adv=0, hay_estandar=False):
    return COMP_BASE_H + n_filas * COMP_ROW_H + n_adv * 22 + (18 if hay_estandar else 0)


def marcador_panel_height(n):
    return 44 + n * MARC_ROW_H + 10


def compute_height(has_market_panel, has_odds_panel, last10_rows=0, n_marcadores=1, rec_alerta=False, has_cover_panel=False, n_comparacion=0,
                   n_adv=0, hay_estandar=False, partido_jugado=False, n_cover_notas=0, n_adv_momios=0):
    y = 140
    y += 130 + 20  # forma reciente
    if last10_rows:
        y += last10_panel_height(last10_rows) + 20
    y += 130 + 20  # probabilidad
    y += marcador_panel_height(n_marcadores) + 20  # marcador esperado
    y += (PLAYED_H if partido_jugado else REC_H + (REC_ALERT_EXTRA if rec_alerta else 0)) + 20   # recomendacion
    if has_market_panel:
        y += EDGE_H + 20
    if has_cover_panel:
        y += cobertura_height(n_cover_notas) + 20
    if n_comparacion:
        y += comparacion_height(n_comparacion, n_adv, hay_estandar) + 20
    if has_odds_panel:
        y += ODDS_H + ODDS_ADV_H * n_adv_momios + 20
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


def juego_ya_jugado(linea, last10):
    """True si el partido de esa linea ya se jugo, segun el historial del cache.

    Un partido esta jugado si aparece en los ultimos 10 de alguno de sus equipos, o si
    es anterior al partido mas antiguo de esa ventana (todo lo previo ya se jugo).
    Si la linea trae el campo "played" se usa tal cual.
    """
    if "played" in linea:
        return bool(linea["played"])
    clave = (linea["season"], linea["week"])
    for team, opp, ha in ((linea["home_team"], linea["away_team"], "local"),
                          (linea["away_team"], linea["home_team"], "visitante")):
        games = last10.get(team) or []
        if not games:
            continue
        if any((g["season"], g["week"]) == clave and g["opponent"] == opp and g["home_away"] == ha
               for g in games):
            return True
        if clave < min((g["season"], g["week"]) for g in games):
            return True
    return False


def find_latest_market_line(market_lines, home, away, last10=None):
    """Busca, entre las lineas guardadas, la mas reciente (mayor season/week)
    para ese cruce exacto (local/visitante en ese orden) cuyo partido NO se haya
    jugado todavia. Nunca devuelve la linea de un partido ya jugado.

    `last10` (historial del cache) es lo que permite saber que partidos ya se
    jugaron; sin el no se puede verificar y se devuelve la mas reciente (modo
    antiguo, solo para compatibilidad)."""
    candidatos = [m for m in market_lines if m["home_team"] == home and m["away_team"] == away]
    if last10 is not None:
        candidatos = [m for m in candidatos if not juego_ya_jugado(m, last10)]
    if not candidatos:
        return None, None, None
    mejor = max(candidatos, key=lambda m: (m["season"], m["week"]))
    return mejor["spread_line"], mejor["total_line"], (mejor["season"], mejor["week"])


def resolver_partido(market_lines, home, away, last10):
    """Estado del partido `home` (local) vs `away` segun el cache.

    - "pendiente": hay una linea de un partido aun no jugado (spread, total, semana).
    - "jugado": no hay linea pendiente y el ultimo cruce con linea ya se jugo y su
      resultado esta en los ultimos 10 del local -> trae season, week y marcador.
    - "sin_linea": no hay linea para ese cruce (o su resultado ya salio del historial).
    """
    spread, total, sw = find_latest_market_line(market_lines, home, away, last10)
    if sw is not None:
        return {"estado": "pendiente", "spread_line": spread, "total_line": total, "season_week": sw}
    cruces = [m for m in market_lines if m["home_team"] == home and m["away_team"] == away]
    for m in sorted(cruces, key=lambda m: (m["season"], m["week"]), reverse=True):
        for g in last10.get(home, []):
            if (g["season"], g["week"]) == (m["season"], m["week"]) and g["opponent"] == away \
                    and g["home_away"] == "local":
                return {"estado": "jugado", "season": m["season"], "week": m["week"],
                        "home_score": g["team_score"], "away_score": g["opp_score"]}
    return {"estado": "sin_linea"}


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

    # advertencias sobre los datos capturados
    margen = margen_de_casa(ml_home, ml_away)
    advertencias = []
    if margen < 0:
        advertencias.append(texto_margen_negativo("ganador"))
    if max(ev_home, ev_away, ev_home_modelo, ev_away_modelo) > VE_INUSUAL:
        advertencias.append(TEXTO_VE_ALTO)

    return {
        "advertencias": advertencias,
        "margen": margen,
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


def subtitulo(pred, season_week=None, jugado=None):
    """Subtitulo de la ficha; refleja la configuracion REAL (EPA, metodo, linea)."""
    puntos = "POISSON + EPA" if pred.get("usa_epa") else "POISSON (SIN EPA)"
    base = f"PROYECCION NFL  ·  PUNTOS: {puntos}  ·  PROB: {pred.get('metodo', 'Poisson')}"
    if jugado:
        return f"PROYECCION NFL  ·  PARTIDO YA JUGADO (T{jugado['season']} S{jugado['week']})  ·  PUNTOS: {puntos}  ·  PROB: {pred.get('metodo', 'Poisson')}"
    if pred.get("market_used") and season_week:
        return f"{base}  ·  LINEA T{season_week[0]} S{season_week[1]}"
    return f"{base}  ·  SIN LINEA DE MERCADO"


def build_image(home, away, pred, rec, teams, season_week=None, momios=None, last10=None, cobertura=None, comparacion=None,
                jugado=None):
    last10 = last10 or {}
    if jugado:  # partido ya jugado: ni recomendacion ni VE
        rec = None
        momios = cobertura = comparacion = None
    home_games = last10.get(home, [])
    away_games = last10.get(away, [])
    last10_rows = min(10, max(len(home_games), len(away_games)))

    has_market_panel = bool(pred.get("market_used") and pred.get("edge_home") is not None)
    has_odds_panel = momios is not None
    has_cover_panel = bool(cobertura)
    cover_notas = cobertura_notas(cobertura) if cobertura else []
    n_adv_momios = len(momios["advertencias"]) if momios else 0
    n_comp = len(comparacion["filas"]) if comparacion else 0
    n_adv = len(comparacion["advertencias"]) if comparacion else 0
    hay_estandar = bool(comparacion and any(f["cuota_estandar"] for f in comparacion["filas"]))
    filas_marc = marcadores(pred)
    H = compute_height(has_market_panel, has_odds_panel, last10_rows,
                       len(filas_marc), bool(rec and rec.get("alerta_texto")), has_cover_panel, n_comp,
                       n_adv, hay_estandar, bool(jugado), len(cover_notas), n_adv_momios)

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

    subt = subtitulo(pred, season_week, jugado)
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
        w = pred.get("market_weight", 0.0)
        titulo_prob = ("Probabilidad de resultado · MERCADO (peso 100%)" if w >= 0.999
                       else f"Probabilidad de resultado · MEZCLA ({mezcla_texto(w)})")
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

    # ---- Panel: marcador esperado + total (cada fila etiquetada; total = suma de lo mostrado) ----
    p3_h = marcador_panel_height(len(filas_marc))
    panel(d, [margin, y, margin + content_w, y + p3_h])
    section_label(d, margin + 24, y + 16, "Marcador esperado")
    right_x = margin + content_w - 24
    for k, fila in enumerate(filas_marc):
        ry = y + 40 + k * MARC_ROW_H
        d.text((margin + 24, ry), fila["etiqueta"], font=font(11, bold=True), fill=MUTED)
        marcador = f"{home}  {fila['home']}   -   {fila['away']}  {away}"
        d.text((margin + 24, ry + 16), marcador, font=font(26, bold=True), fill=FG)
        tw = text_w(d, fila["total_etiqueta"], font(11, bold=True))
        total_val = f"{fila['total']} pts"
        vw = text_w(d, total_val, font(22, bold=True))
        d.text((right_x - tw, ry), fila["total_etiqueta"], font=font(11, bold=True), fill=MUTED)
        d.text((right_x - vw, ry + 18), total_val, font=font(22, bold=True), fill=FG)

    y += p3_h + 20

    # ---- Panel: recomendacion (favorito y valor por separado) / partido ya jugado ----
    if jugado:
        panel(d, [margin, y, margin + content_w, y + PLAYED_H], fill=(30, 26, 13), outline=(110, 90, 30))
        section_label(d, margin + 24, y + 16, "Partido ya jugado")
        res = (f"Partido ya jugado: resultado {home} {jugado['home_score']:.0f} - "
               f"{jugado['away_score']:.0f} {away}")
        d.text((margin + 24, y + 42), res, font=font(22, bold=True), fill=WARN)
        d.text((margin + 24, y + 74), "Sin recomendacion ni VE para un partido que ya se jugo.",
               font=font(12), fill=MUTED)
        y += PLAYED_H + 20
    else:
        p4_h = REC_H + (REC_ALERT_EXTRA if rec.get("alerta_texto") else 0)
        panel(d, [margin, y, margin + content_w, y + p4_h], fill=(13, 30, 25), outline=(0, 90, 60))
        section_label(d, margin + 24, y + 16, "Recomendacion")
        d.text((margin + 24, y + 40), rec["favorito_texto"], font=font(21, bold=True), fill=FG)
        color_valor = WARN if rec.get("valor_lado") else MUTED
        d.text((margin + 24, y + 72), rec["valor_texto"], font=font(21, bold=True), fill=color_valor)
        if rec.get("alerta_texto"):
            d.text((margin + 24, y + 104), rec["alerta_texto"], font=font(14, bold=True), fill=NEG)
        y += p4_h + 20

    # ---- Panel: valor vs mercado (edge) ----
    if not jugado and pred.get("market_used") and pred.get("edge_home") is not None:
        p5_h = EDGE_H
        panel(d, [margin, y, margin + content_w, y + p5_h])
        section_label(d, margin + 24, y + 16, "Diferencia modelo vs mercado (no validada)")

        edge_home, edge_away = pred["edge_home"], pred["edge_away"]
        edge_color = WARN if rec.get("valor_lado") else MUTED

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
                 ACCENT if diff >= UMBRAL_VALOR else (NEG if diff <= -UMBRAL_VALOR else MUTED)),
            ):
                d.text((cx - text_w(d, txt, font(16, bold=True)), ry), txt, font=font(16, bold=True), fill=color)

        veredicto = rec["valor_texto"]
        d.text((x_left, y + 112), veredicto, font=font(15, bold=True), fill=edge_color)

        y += p5_h + 20

    # ---- Panel: spread y total (tu casa o consenso, nunca mezclados) · modelo puro ----
    if cobertura:
        c_h = cobertura_height(len(cover_notas))
        panel(d, [margin, y, margin + content_w, y + c_h])
        titulo = ("Spread y total vs tu casa · MODELO PURO (no validado)" if cobertura["usa_casa"]
                  else "Spread y total vs consenso · MODELO PURO (no validado)")
        section_label(d, margin + 24, y + 16, titulo)
        x_left = margin + 24
        right = margin + content_w - 24
        cols = (right - 380, right - 270, right - 150, right)
        for cx, head in zip(cols, ("PROB. MODELO", "PUSH", "CUOTA", "EV POR $100")):
            d.text((cx - text_w(d, head, font(11, bold=True)), y + 42), head, font=font(11, bold=True), fill=MUTED)
        for i, fila in enumerate(cobertura["filas"]):
            ry = y + 62 + i * 24 + (8 if i >= 2 else 0)
            etiqueta = fila["etiqueta"].format(home=home, away=away)
            if cobertura["mixto"]:
                etiqueta += "  (tu casa)" if fila["fuente"] == "casa" else "  (consenso)"
            d.text((x_left, ry), etiqueta, font=font(16, bold=True), fill=FG)
            cuota_txt = f"{fila['cuota']:+.0f}" + ("*" if fila["cuota_estandar"] else "")
            ev = fila["ev"]
            for cx, txt, color in (
                (cols[0], f"{fila['prob']}%", FG),
                (cols[1], f"{fila['p_push']}%" if fila["p_push"] >= 0.05 else "-", MUTED),
                (cols[2], cuota_txt, MUTED),
                (cols[3], f"{ev:+.2f}", ACCENT if ev > 0 else (NEG if ev < 0 else MUTED)),
            ):
                d.text((cx - text_w(d, txt, font(16, bold=True)), ry), txt, font=font(16, bold=True), fill=color)
        for i, nota in enumerate(cover_notas):
            d.text((x_left, y + 170 + i * COVER_NOTE_H), nota, font=font(11), fill=MUTED)
        y += c_h + 20

    # ---- Panel: comparacion de lineas (tu casa vs consenso) ----
    if comparacion:
        advs = comparacion["advertencias"]
        p_h = comparacion_height(n_comp, len(advs), hay_estandar)
        aw = 22 * len(advs)
        panel(d, [margin, y, margin + content_w, y + p_h])
        section_label(d, margin + 24, y + 16, "Comparacion de lineas · tu casa vs consenso")
        x_left = margin + 24
        c = comparacion
        cons = f"Consenso: {home} {-c['consenso_spread']:+.1f}  ·  total {c['consenso_total']:g}"
        if c["casa_distinta_spread"] or c["casa_distinta_total"]:
            casa_txt = f"Tu casa: {home} {-c['casa_x_spread']:+.1f}  ·  total {c['casa_total']:g}"
        else:
            casa_txt = "Tu casa: mismas lineas que el consenso"
        d.text((x_left, y + 40), cons + "     |     " + casa_txt, font=font(14, bold=True), fill=FG)
        d.text((x_left, y + 62), "El consenso del mercado se toma como probabilidad verdadera.",
               font=font(11), fill=MUTED)
        for i, adv in enumerate(advs):
            d.text((x_left, y + 84 + i * 22), "! " + adv["texto"], font=font(13, bold=True), fill=WARN)
        cols = (x_left + 420, x_left + 530, x_left + 620, x_left + 740, margin + content_w - 24)
        heads = ("CUOTA", "P GANA", "PUSH", "CUOTA JUSTA", "VE POR $100")
        for cx, head in zip(cols, heads):
            d.text((cx - text_w(d, head, font(11, bold=True)), y + 86 + aw), head, font=font(11, bold=True), fill=MUTED)
        for i, f in enumerate(c["filas"]):
            ry = y + 106 + aw + i * COMP_ROW_H
            etq = f["etiqueta"].format(home=home, away=away)
            if f["linea"]:
                etq = f"{etq} {f['linea']}"
            d.text((x_left, ry), etq, font=font(16, bold=True), fill=FG)
            justa = f["cuota_justa"]
            for cx, txt, color in (
                (cols[0], f"{f['cuota']:+.0f}" + ("*" if f["cuota_estandar"] else ""), MUTED),
                (cols[1], f"{f['p_gana']:.1f}%", FG),
                (cols[2], f"{f['p_push']:.1f}%" if f["p_push"] >= 0.05 else "-", MUTED),
                (cols[3], f"{justa:+.0f}" if justa else "-", MUTED),
                (cols[4], f"{f['ev']:+.2f}", ACCENT if f["ev"] > 0 else (NEG if f["ev"] < 0 else MUTED)),
            ):
                d.text((cx - text_w(d, txt, font(16, bold=True)), ry), txt, font=font(16, bold=True), fill=color)
        mc = c["margen_casa"]
        partes = [f"{k} {v:.1f}%" for k, v in (("spread", mc["spread"]), ("total", mc["total"]),
                                                ("ganador", mc["ganador"])) if v is not None]
        nota = (f"Frecuencias nflverse 2015-2025 con linea cercana (n ~{c['n_ef_spread']} / ~{c['n_ef_total']})."
                + (f"  Margen casa: {', '.join(partes)}." if partes else ""))
        d.text((x_left, y + p_h - 28), nota, font=font(11), fill=MUTED)
        if hay_estandar:
            d.text((x_left, y + p_h - 46), "* cuota por defecto -110 (no capturada); sin asterisco = la capturaste tu.",
                   font=font(11), fill=MUTED)
        y += p_h + 20

    # ---- Panel: tus momios (valor real con vig removido) ----
    if momios:
        p6_h = ODDS_H + ODDS_ADV_H * len(momios["advertencias"])
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
                (right, f"CON BARRA ({mezcla_texto(pred.get('market_weight', 0.0)).upper()})", "ev_home", "ev_away"),
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
                   "La barra ya incluye al mercado: su EV es circular. El EV del modelo puro es la señal independiente (no validada).",
                   font=font(11), fill=MUTED)

        for i, adv in enumerate(momios["advertencias"]):
            d.text((x_left, y + ODDS_H + 4 + i * ODDS_ADV_H), "! " + adv, font=font(13, bold=True), fill=WARN)

        y += p6_h + 20

    d.text((margin, H - 34), "Uso informativo — no es asesoria financiera",
           font=font(12), fill=MUTED)

    return img


def parse_lineas_casa(qs):
    """Lineas de tu casa (opcionales): sp_linea = linea del LOCAL en formato casa
    (-2.5 = local favorito por 2.5; +3 = local underdog), total_linea = total de la casa."""
    out = {}
    for k in ("sp_linea", "total_linea"):
        raw = (qs.get(k) or [""])[0].strip()
        if raw:
            try:
                out[k] = float(raw)
            except ValueError:
                pass
    return out


def construir_comparacion(spread_line, total_line, odds, momios, lineas_casa, equipo_local="LOCAL"):
    """Comparacion de tu casa contra el consenso, o None si no hay linea de consenso
    (partido sin linea pendiente) o no capturaste ninguna cuota."""
    if spread_line is None or total_line is None:
        return None
    cuotas = dict(odds or {})
    if momios:
        cuotas["ml_home"], cuotas["ml_away"] = momios["ml_home"], momios["ml_away"]
    lineas_casa = lineas_casa or {}
    return comparar_casa(spread_line, total_line, cuotas,
                         lineas_casa.get("sp_linea"), lineas_casa.get("total_linea"),
                         equipo_local=equipo_local)


def parse_odds(qs):
    """Cuotas americanas opcionales del spread y del total: sp_home, sp_away, over, under."""
    odds = {}
    for k in ("sp_home", "sp_away", "over", "under"):
        raw = (qs.get(k) or [""])[0].strip()
        if raw:
            try:
                odds[k] = float(raw)
            except ValueError:
                pass
    return odds


def preparar_ficha(home, away, cache, odds=None, lineas_casa=None, ml_home=None, ml_away=None):
    """Todo lo que necesita build_image, igual para la API y para local_test.py.
    Devuelve (kwargs_para_build_image, advertencias_de_texto)."""
    teams, league_avg_pts, market_lines, last10 = cache
    estado = resolver_partido(market_lines, home, away, last10)
    model_pred = predict_match(home, away, stats_df_like(teams), league_avg_pts)
    if estado["estado"] == "pendiente":
        spread_line, total_line, season_week = (estado["spread_line"], estado["total_line"],
                                                estado["season_week"])
    else:
        spread_line = total_line = season_week = None
    pred = blend_prediction(model_pred, spread_line, total_line)

    jugado = estado if estado["estado"] == "jugado" else None
    rec = cobertura = momios = comparacion = None
    if not jugado:
        rec = recomendacion_apuesta(pred, home, away)
        cobertura = analizar_spread_total(pred, odds or {}, lineas_casa or {})
        if ml_home is not None and ml_away is not None:
            momios = analizar_momios(pred, ml_home, ml_away)
        comparacion = construir_comparacion(spread_line, total_line, odds, momios, lineas_casa, home)
    advertencias = [a["texto"] for a in comparacion["advertencias"]] if comparacion else []
    if momios:
        advertencias += [a for a in momios["advertencias"] if a not in advertencias]
    return dict(home=home, away=away, pred=pred, rec=rec, teams=teams, season_week=season_week,
                momios=momios, last10=last10, cobertura=cobertura, comparacion=comparacion,
                jugado=jugado), advertencias


def _float_o_none(raw):
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


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

        kwargs, advertencias = preparar_ficha(
            home, away, (teams, league_avg_pts, market_lines, last10_games),
            parse_odds(qs), parse_lineas_casa(qs), _float_o_none(ml_home_raw), _float_o_none(ml_away_raw))
        for texto in advertencias:
            print(f"ADVERTENCIA {home} vs {away}: {texto}", file=sys.stderr)
        img = build_image(**kwargs)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        self.wfile.write(buf.getvalue())
