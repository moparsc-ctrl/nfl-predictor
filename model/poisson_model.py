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


def recomendacion_apuesta(pred: dict, umbral_favorito: float = 62.0):
    """Lógica simple de recomendación tipo doble oportunidad, adaptable."""
    p_home, p_away = pred["p_home_win"], pred["p_away_win"]
    favorito, prob = ("local", p_home) if p_home >= p_away else ("visitante", p_away)

    if prob >= umbral_favorito:
        return f"Gana {favorito} ({prob:.1f}%)"
    else:
        # partido parejo: no hay pick claro de ganador directo
        margen = abs(p_home - p_away)
        if margen < 8:
            return "Partido parejo — evitar apuesta a ganador directo"
        return f"Leve favorito: {favorito} ({prob:.1f}%) — apuesta con cautela"
