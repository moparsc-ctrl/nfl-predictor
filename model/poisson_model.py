"""
Modelo Poisson validado en el notebook: 64.6% de acierto en ganador,
~10.5 pts de error promedio en el total (temporada 2025).
"""
from statistics import NormalDist

from scipy.stats import poisson

from model import config as cfg

_ND = NormalDist()


def predict_match(home_team, away_team, stats_df, league_avg_pts,
                   home_field_boost=1.05, use_epa=None, max_pts=60,
                   method=None, sigma_margen=None, sigma_total=None, total_bias=None):
    """Poisson + EPA estima los puntos esperados; el metodo decide como pasar de
    puntos a probabilidad de victoria ("normal" por defecto, ver model/config.py)."""
    method = method or cfg.METODO_PROB
    sigma_margen = cfg.SIGMA_MARGEN_MODELO if sigma_margen is None else sigma_margen
    sigma_total = cfg.SIGMA_TOTAL if sigma_total is None else sigma_total
    total_bias = cfg.TOTAL_BIAS if total_bias is None else total_bias
    use_epa = cfg.USAR_EPA if use_epa is None else use_epa

    off_col = "off_strength_epa" if use_epa and "off_strength_epa" in stats_df.columns else "off_strength"
    def_col = "def_strength_epa" if use_epa and "def_strength_epa" in stats_df.columns else "def_strength"

    home_off = stats_df.loc[home_team, off_col]
    away_def = stats_df.loc[away_team, def_col]
    away_off = stats_df.loc[away_team, off_col]
    home_def = stats_df.loc[home_team, def_col]

    lambda_home = league_avg_pts * home_off * away_def * home_field_boost
    lambda_away = league_avg_pts * away_off * home_def

    # correccion del sesgo del total, repartida: el margen no cambia
    lambda_home = max(lambda_home + total_bias / 2, 0.5)
    lambda_away = max(lambda_away + total_bias / 2, 0.5)

    margen = lambda_home - lambda_away
    total_esperado = lambda_home + lambda_away

    if method == "poisson":
        p_home_win, p_away_win = _poisson_win_probs(lambda_home, lambda_away, max_pts)
        metodo_txt = "Poisson"
    else:
        p_home_win = prob_victoria_normal(margen, sigma_margen)
        p_away_win = 1 - p_home_win
        metodo_txt = f"Normal σ={sigma_margen:g}"

    return {
        "lambda_home": round(lambda_home, 1),
        "lambda_away": round(lambda_away, 1),
        "p_home_win": round(p_home_win * 100, 1),
        "p_away_win": round(p_away_win * 100, 1),
        "total_esperado_puntos": round(total_esperado, 1),
        "margen_esperado": round(margen, 2),
        "metodo": metodo_txt,
        "sigma_margen": sigma_margen,
        "sigma_total": sigma_total,
    }


def prob_victoria_normal(margen_esperado: float, sigma: float = None) -> float:
    """P(local gana) = Phi(margen_esperado / sigma). Devuelve 0-1."""
    sigma = cfg.SIGMA_MARGEN_MODELO if sigma is None else sigma
    return _ND.cdf(margen_esperado / sigma)


def _poisson_win_probs(lambda_home, lambda_away, max_pts=60):
    home_probs = [poisson.pmf(i, lambda_home) for i in range(max_pts)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(max_pts)]
    p_home_win = p_away_win = p_tie = 0.0
    for i in range(max_pts):
        for j in range(max_pts):
            p = home_probs[i] * away_probs[j]
            if i > j:
                p_home_win += p
            elif i < j:
                p_away_win += p
            else:
                p_tie += p
    # Reparte el "empate" simulando tiempo extra, segun fuerza relativa
    home_edge = lambda_home / (lambda_home + lambda_away)
    p_home_win += p_tie * home_edge
    p_away_win += p_tie * (1 - home_edge)
    # la truncacion en max_pts pierde masa con lambdas altas: normaliza para que sume 1
    tot = p_home_win + p_away_win
    return p_home_win / tot, p_away_win / tot


# Edge minimo (puntos de probabilidad, modelo puro - mercado) para hablar de "valor".
# Lo usan la recomendacion y el panel de edge, para que nunca se contradigan.
UMBRAL_VALOR = cfg.UMBRAL_VALOR
# Edge a partir del cual es mas probable que el modelo exagere que que haya valor real.
EDGE_ALTO = cfg.EDGE_ALTO


def recomendacion_apuesta(pred: dict, home: str, away: str,
                          umbral_favorito: float = 62.0,
                          umbral_valor: float = UMBRAL_VALOR,
                          edge_alto: float = EDGE_ALTO) -> dict:
    """Separa dos preguntas distintas:

    - FAVORITO: quien gana segun la probabilidad mostrada (la mezcla con el
      mercado si hay linea; casi siempre coincide con el favorito de la casa).
    - VALOR: donde el modelo puro ve mas probabilidad que el mercado (edge).
      Es lo que interesa para apostar y puede apuntar al otro equipo.
    """
    p_home, p_away = pred["p_home_win"], pred["p_away_win"]
    lado_fav = "home" if p_home >= p_away else "away"
    equipo_fav = home if lado_fav == "home" else away
    prob = max(p_home, p_away)

    if prob >= umbral_favorito:
        favorito_texto = f"Favorito: {equipo_fav} ({prob:.1f}%)"
    elif abs(p_home - p_away) < 8:
        favorito_texto = f"Partido parejo — favorito marginal: {equipo_fav} ({prob:.1f}%)"
    else:
        favorito_texto = f"Favorito: {equipo_fav} ({prob:.1f}%) — apuesta con cautela"

    edge_h, edge_a = pred.get("edge_home"), pred.get("edge_away")
    valor_lado, valor_edge, alerta_texto = None, None, None
    if edge_h is None or edge_a is None:
        valor_texto = "Sin linea de mercado para comparar"
    else:
        lado, edge = ("home", edge_h) if edge_h >= edge_a else ("away", edge_a)
        if edge >= umbral_valor:
            valor_lado, valor_edge = lado, edge
            equipo_valor = home if lado == "home" else away
            if cfg.EDGE_VALIDADO is not None and edge >= cfg.EDGE_VALIDADO:
                valor_texto = f"Valor: {equipo_valor} (+{edge:.1f} pts vs mercado)"
            else:
                valor_texto = f"Diferencia modelo vs mercado (no validada): {equipo_valor} +{edge:.1f} pts"
            if edge >= edge_alto:
                alerta_texto = "Edge muy grande: revisar, el modelo puro suele exagerar"
        else:
            valor_texto = "Sin diferencia relevante modelo vs mercado"

    return {
        "favorito_lado": lado_fav,
        "favorito_texto": favorito_texto,
        "valor_lado": valor_lado,
        "valor_edge": valor_edge,
        "valor_texto": valor_texto,
        "alerta_texto": alerta_texto,
    }
