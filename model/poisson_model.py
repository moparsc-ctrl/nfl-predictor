"""
Modelo Poisson validado en el notebook: 64.6% de acierto en ganador,
~10.5 pts de error promedio en el total (temporada 2025).
"""
from scipy.stats import poisson


def predict_match(home_team, away_team, stats_df, league_avg_pts,
                   home_field_boost=1.05, use_epa=True, max_pts=60):
    off_col = "off_strength_epa" if use_epa and "off_strength_epa" in stats_df.columns else "off_strength"
    def_col = "def_strength_epa" if use_epa and "def_strength_epa" in stats_df.columns else "def_strength"

    home_off = stats_df.loc[home_team, off_col]
    away_def = stats_df.loc[away_team, def_col]
    away_off = stats_df.loc[away_team, off_col]
    home_def = stats_df.loc[home_team, def_col]

    lambda_home = league_avg_pts * home_off * away_def * home_field_boost
    lambda_away = league_avg_pts * away_off * home_def

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

    # Reparte el "empate" simulando tiempo extra, según fuerza relativa
    home_edge = lambda_home / (lambda_home + lambda_away)
    p_home_win += p_tie * home_edge
    p_away_win += p_tie * (1 - home_edge)

    total_esperado = lambda_home + lambda_away

    return {
        "lambda_home": round(lambda_home, 1),
        "lambda_away": round(lambda_away, 1),
        "p_home_win": round(p_home_win * 100, 1),
        "p_away_win": round(p_away_win * 100, 1),
        "total_esperado_puntos": round(total_esperado, 1),
    }


# Edge minimo (puntos de probabilidad, modelo puro - mercado) para hablar de "valor".
# Lo usan la recomendacion y el panel de edge, para que nunca se contradigan.
UMBRAL_VALOR = 3.0
# Edge a partir del cual es mas probable que el modelo exagere que que haya valor real.
EDGE_ALTO = 15.0


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
        valor_texto = "Valor: sin linea de mercado para comparar"
    else:
        lado, edge = ("home", edge_h) if edge_h >= edge_a else ("away", edge_a)
        if edge >= umbral_valor:
            valor_lado, valor_edge = lado, edge
            equipo_valor = home if lado == "home" else away
            valor_texto = f"Valor segun el modelo: {equipo_valor} (+{edge:.1f} pts vs mercado)"
            if edge >= edge_alto:
                alerta_texto = "Edge muy grande: revisar, el modelo puro suele exagerar"
        else:
            valor_texto = "Sin valor claro"

    return {
        "favorito_lado": lado_fav,
        "favorito_texto": favorito_texto,
        "valor_lado": valor_lado,
        "valor_edge": valor_edge,
        "valor_texto": valor_texto,
        "alerta_texto": alerta_texto,
    }
