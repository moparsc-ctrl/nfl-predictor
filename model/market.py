"""
Convierte lineas de mercado (spread_line, total_line) a probabilidad
y las mezcla con la salida del modelo Poisson, para anclar predicciones
extremas a algo mas cercano a lo que realmente se mueve en las casas.

Tambien incluye utilidades para detectar "valor" (edge) comparando la
probabilidad del modelo contra la probabilidad implicita del mercado.
"""
import math
from statistics import NormalDist

SIGMA_MARGEN = 13.86
_ND = NormalDist()


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def spread_to_home_prob(spread_line: float) -> float:
    """Probabilidad (0-1) de que gane el LOCAL segun el spread.

    Convencion nflverse: spread_line > 0 significa que el LOCAL es favorito
    (verificado contra margenes reales; ver tests/test_market.py).
    """
    return _norm_cdf(spread_line / SIGMA_MARGEN)


def blend_prediction(model_pred: dict, spread_line, total_line, market_weight: float = 0.65) -> dict:
    if spread_line is None or total_line is None:
        out = dict(model_pred)
        out["market_used"] = False
        out["edge_home"] = None
        out["edge_away"] = None
        return out

    market_home_prob = spread_to_home_prob(spread_line) * 100
    market_away_prob = 100 - market_home_prob

    p_home = market_weight * market_home_prob + (1 - market_weight) * model_pred["p_home_win"]
    p_away = market_weight * market_away_prob + (1 - market_weight) * model_pred["p_away_win"]
    total_p = p_home + p_away
    p_home = p_home / total_p * 100
    p_away = p_away / total_p * 100

    total_pts = market_weight * total_line + (1 - market_weight) * model_pred["total_esperado_puntos"]

    edge_home = round(model_pred["p_home_win"] - market_home_prob, 1)
    edge_away = round(model_pred["p_away_win"] - market_away_prob, 1)

    return {
        "lambda_home": model_pred["lambda_home"],
        "lambda_away": model_pred["lambda_away"],
        "p_home_win": round(p_home, 1),
        "p_away_win": round(p_away, 1),
        "total_esperado_puntos": round(total_pts, 1),
        "market_used": True,
        "market_spread": spread_line,
        "market_total": total_line,
        "market_home_prob": round(market_home_prob, 1),
        "market_away_prob": round(market_away_prob, 1),
        "edge_home": edge_home,
        "edge_away": edge_away,
        "model_home_prob": model_pred["p_home_win"],
        "model_away_prob": model_pred["p_away_win"],
        # totales por separado: total_esperado_puntos es la MEZCLA
        "total_modelo": model_pred["total_esperado_puntos"],
        "total_mercado": total_line,
        "market_weight": market_weight,
    }


def marcadores(pred: dict) -> list:
    """Filas de marcador esperado que muestra la ficha, cada una con su etiqueta.

    - "puro": puntos del Poisson (modelo puro). Su ganador coincide con la
      probabilidad del modelo puro.
    - "ajustado" (solo si hay linea de mercado): total mezclado + margen que
      implica la probabilidad mezclada de la barra (margen = sigma * Phi^-1(p_local)).
      Suma el total mezclado y su ganador coincide con la barra.

    En cada fila el total es la suma de los puntos mostrados (ya redondeados),
    para que siempre cuadre a la vista.
    """
    filas = []

    h = round(float(pred["lambda_home"]), 1)
    a = round(float(pred["lambda_away"]), 1)
    p_puro = pred.get("model_home_prob", pred["p_home_win"])
    filas.append({
        "clave": "puro",
        "etiqueta": "MODELO PURO",
        "total_etiqueta": "TOTAL (OVER/UNDER) · MODELO PURO",
        "home": h, "away": a, "total": round(h + a, 1),
        "p_home": float(p_puro),
    })

    if pred.get("market_used"):
        w = pred.get("market_weight", 0.65)
        mezcla = f"{w * 100:.0f}% mercado + {(1 - w) * 100:.0f}% modelo"
        total = float(pred["total_esperado_puntos"])
        p_home = float(pred["p_home_win"])
        p_clamp = min(max(p_home, 0.1), 99.9) / 100
        margen = SIGMA_MARGEN * _ND.inv_cdf(p_clamp)
        h2 = round(max((total + margen) / 2, 0.0), 1)
        a2 = round(max((total - margen) / 2, 0.0), 1)
        filas.append({
            "clave": "ajustado",
            "etiqueta": f"AJUSTADO A LA MEZCLA ({mezcla})",
            "total_etiqueta": f"TOTAL (OVER/UNDER) · MEZCLA {w * 100:.0f}/{(1 - w) * 100:.0f}",
            "home": h2, "away": a2, "total": round(h2 + a2, 1),
            "p_home": p_home,
        })

    return filas


def remove_vig(prob_a_pct: float, prob_b_pct: float):
    total = prob_a_pct + prob_b_pct
    return round(prob_a_pct / total * 100, 1), round(prob_b_pct / total * 100, 1)


def moneyline_to_implied_prob(moneyline: float) -> float:
    if moneyline > 0:
        return 100 / (moneyline + 100) * 100
    else:
        return -moneyline / (-moneyline + 100) * 100


def expected_value(prob_real_pct: float, moneyline: float, stake: float = 100) -> float:
    p = prob_real_pct / 100
    if moneyline > 0:
        ganancia_si_gana = stake * (moneyline / 100)
    else:
        ganancia_si_gana = stake * (100 / -moneyline)
    return round(p * ganancia_si_gana - (1 - p) * stake, 2)
