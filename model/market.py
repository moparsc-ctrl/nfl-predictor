"""
Convierte lineas de mercado (spread_line, total_line) a probabilidad
y las mezcla con la salida del modelo Poisson, para anclar predicciones
extremas a algo mas cercano a lo que realmente se mueve en las casas.

Tambien incluye utilidades para detectar "valor" (edge) comparando la
probabilidad del modelo contra la probabilidad implicita del mercado.
"""
import math
from statistics import NormalDist

from model import config as cfg

SIGMA_MARGEN = cfg.SIGMA_MARGEN_MERCADO
_ND = NormalDist()


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def spread_to_home_prob(spread_line: float) -> float:
    """Probabilidad (0-1) de que gane el LOCAL segun el spread.

    Convencion nflverse: spread_line > 0 significa que el LOCAL es favorito
    (verificado contra margenes reales; ver tests/test_market.py).
    """
    return _norm_cdf(spread_line / SIGMA_MARGEN)


def blend_prediction(model_pred: dict, spread_line, total_line, market_weight: float = None) -> dict:
    market_weight = cfg.MARKET_WEIGHT if market_weight is None else market_weight
    if spread_line is None or total_line is None:
        out = dict(model_pred)
        out["market_used"] = False
        out["market_weight"] = 0.0
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
        "margen_esperado": model_pred.get("margen_esperado", model_pred["lambda_home"] - model_pred["lambda_away"]),
        "metodo": model_pred.get("metodo", "Poisson"),
        "usa_epa": model_pred.get("usa_epa", False),
        "sigma_margen": model_pred.get("sigma_margen", cfg.SIGMA_MARGEN_MODELO),
        "sigma_total": model_pred.get("sigma_total", cfg.SIGMA_TOTAL),
    }


def mezcla_texto(w: float) -> str:
    """'65% mercado + 35% modelo' (o '100% mercado' si el modelo no pesa)."""
    if w >= 0.999:
        return "100% mercado"
    return f"{w * 100:.0f}% mercado + {(1 - w) * 100:.0f}% modelo"


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
        w = pred.get("market_weight", cfg.MARKET_WEIGHT)
        mezcla = mezcla_texto(w)
        total = float(pred["total_esperado_puntos"])
        p_home = float(pred["p_home_win"])
        p_clamp = min(max(p_home, 0.1), 99.9) / 100
        margen = SIGMA_MARGEN * _ND.inv_cdf(p_clamp)
        h2 = round(max((total + margen) / 2, 0.0), 1)
        a2 = round(max((total - margen) / 2, 0.0), 1)
        filas.append({
            "clave": "ajustado",
            "etiqueta": (f"AJUSTADO A LA MEZCLA ({mezcla})" if w < 0.999
                         else "AJUSTADO AL MERCADO (peso 100%)"),
            "total_etiqueta": (f"TOTAL (OVER/UNDER) · MEZCLA {w * 100:.0f}/{(1 - w) * 100:.0f}" if w < 0.999
                               else "TOTAL (OVER/UNDER) · MERCADO"),
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


# ---------------------------------------------------------------------------
# Cobertura de spread y over/under contra la linea de mercado (modelo puro)
# ---------------------------------------------------------------------------
def _pmf_margen(mu: float, sigma: float, boost: dict):
    ks = range(-70, 71)
    p = {k: _ND.cdf((k + 0.5 - mu) / sigma) - _ND.cdf((k - 0.5 - mu) / sigma) for k in ks}
    if boost:
        p = {k: v * boost.get(abs(k), 1.0) for k, v in p.items()}
    tot = sum(p.values())
    return {k: v / tot for k, v in p.items()}


def prob_cubrir_spread(margen_esperado: float, spread_line: float, sigma: float = None,
                       numeros_clave: bool = None):
    """(P local cubre, P visitante cubre, P push), en 0-1.

    Convencion nflverse: spread_line > 0 = local favorito por esa cantidad; el local
    cubre si margen_real > spread_line. Con numeros_clave=True la distribucion del
    margen (enteros) se pondera en 3, 6, 7, 10 y 14 (ver config.NUMEROS_CLAVE_BOOST).
    """
    sigma = cfg.SIGMA_MARGEN_MODELO if sigma is None else sigma
    numeros_clave = cfg.NUMEROS_CLAVE if numeros_clave is None else numeros_clave
    if numeros_clave:
        pmf = _pmf_margen(margen_esperado, sigma, cfg.NUMEROS_CLAVE_BOOST)
        home = sum(v for k, v in pmf.items() if k > spread_line)
        away = sum(v for k, v in pmf.items() if k < spread_line)
        push = 1.0 - home - away
        return home, away, push
    home = 1 - _ND.cdf((spread_line - margen_esperado) / sigma)
    return home, 1 - home, 0.0


def prob_over(total_esperado: float, total_line: float, sigma_total: float = None):
    """(P over, P under), en 0-1: P(over) = 1 - Phi((linea - total_esperado) / sigma_total)."""
    sigma_total = cfg.SIGMA_TOTAL if sigma_total is None else sigma_total
    over = 1 - _ND.cdf((total_line - total_esperado) / sigma_total)
    return over, 1 - over


def ev_con_push(p_gana: float, p_push: float, american_odds: float, stake: float = 100) -> float:
    """EV por $stake: gana (p_gana), devuelve stake (p_push) o pierde el resto."""
    ganancia = stake * (american_odds / 100) if american_odds > 0 else stake * (100 / -american_odds)
    return round(p_gana * ganancia - (1 - p_gana - p_push) * stake, 2)


def prob_total(total_esperado: float, total_line: float, sigma_total: float = None):
    """(P over, P under, P push), 0-1. Linea de medio punto: sin push. Linea entera: el
    push es la masa de la normal sobre ese total entero (correccion de continuidad)."""
    sigma_total = cfg.SIGMA_TOTAL if sigma_total is None else sigma_total
    if float(total_line).is_integer():
        sup = _ND.cdf((total_line + 0.5 - total_esperado) / sigma_total)
        inf = _ND.cdf((total_line - 0.5 - total_esperado) / sigma_total)
        return 1 - sup, inf, sup - inf
    over = 1 - _ND.cdf((total_line - total_esperado) / sigma_total)
    return over, 1 - over, 0.0


def analizar_spread_total(pred: dict, odds: dict = None, lineas_casa: dict = None):
    """Probabilidad del MODELO PURO de cubrir el spread y de over/under, con EV por $100.

    Cada mercado usa UNA sola fuente, nunca mezcla la linea de una con las cuotas de otra:
      - sp_linea / total_linea capturadas (formato casa; sp_linea = linea del LOCAL):
        se evalua MI linea con MIS cuotas (sp_home, sp_away / over, under); la cuota que
        falte va a -110 marcada como estandar.
      - sin linea capturada: linea del CONSENSO con cuotas -110 estandar (si capturaste
        cuotas de ese mercado sin su linea NO se aplican al consenso; queda una nota).
    El push en lineas enteras usa los numeros clave (3, 6, 7, 10, 14) del spread y la
    normal discretizada en el total.
    Devuelve None si no hay con que evaluar (sin consenso y sin lineas tuyas).
    """
    odds = {k: v for k, v in (odds or {}).items() if v is not None}
    lineas = lineas_casa or {}
    consenso = bool(pred.get("market_used"))
    sp_linea, tot_linea = lineas.get("sp_linea"), lineas.get("total_linea")
    if not consenso and sp_linea is None and tot_linea is None:
        return None

    mu = pred.get("margen_esperado", pred["lambda_home"] - pred["lambda_away"])
    total_modelo = pred.get("total_modelo", pred["total_esperado_puntos"])
    filas, notas = [], []

    def fila(clave, etiqueta, mercado, fuente, linea, p, push, key):
        o = odds.get(key) if fuente == "casa" else None   # cuotas tuyas solo sobre tu linea
        cuota = o if o is not None else -110.0
        filas.append({"clave": clave, "etiqueta": etiqueta, "mercado": mercado, "fuente": fuente,
                      "linea": linea, "prob": round(p * 100, 1), "p_push": round(push * 100, 1),
                      "cuota": cuota, "cuota_estandar": o is None, "ev": ev_con_push(p, push, cuota)})

    # ---- spread
    if sp_linea is not None or consenso:
        if sp_linea is not None:
            fuente, x = "casa", -sp_linea                      # el local cubre si margen > x
            linea_local = sp_linea
        else:
            fuente, x = "consenso", pred["market_spread"]
            linea_local = -x
            if {"sp_home", "sp_away"} & set(odds):
                notas.append("Cuotas del spread sin sp_linea: no se aplican a la linea del consenso.")
        p_h, p_a, p_push = prob_cubrir_spread(mu, x, pred.get("sigma_margen"), numeros_clave=True)
        fila("sp_home", f"{{home}} {linea_local:+.1f}", "spread", fuente, linea_local, p_h, p_push, "sp_home")
        fila("sp_away", f"{{away}} {-linea_local:+.1f}", "spread", fuente, -linea_local, p_a, p_push, "sp_away")

    # ---- total
    if tot_linea is not None or consenso:
        if tot_linea is not None:
            fuente, tl = "casa", tot_linea
        else:
            fuente, tl = "consenso", pred["market_total"]
            if {"over", "under"} & set(odds):
                notas.append("Cuotas del total sin total_linea: no se aplican a la linea del consenso.")
        p_o, p_u, p_pt = prob_total(total_modelo, tl, pred.get("sigma_total"))
        fila("over", f"Over {tl:g}", "total", fuente, tl, p_o, p_pt, "over")
        fila("under", f"Under {tl:g}", "total", fuente, tl, p_u, p_pt, "under")

    return {"filas": filas, "notas": notas,
            "usa_casa": any(f["fuente"] == "casa" for f in filas),
            "mixto": len({f["fuente"] for f in filas}) > 1}
