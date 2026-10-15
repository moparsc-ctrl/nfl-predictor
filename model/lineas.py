"""Comparacion de las lineas de TU casa contra la linea de consenso del mercado.

Idea: el mercado (linea de consenso) es la probabilidad "verdadera". De su spread y
total se obtiene una distribucion de resultados con FRECUENCIAS HISTORICAS
(data/history.json, nflverse 2015-2025): los partidos cuya linea estaba cerca de la
actual, con su margen / total real. Asi el empate exacto (push) en 3, 7, etc. sale de
cuantas veces ocurrio de verdad, no de una normal.

Convencion nflverse: spread_line > 0 = el local es favorito; el local cubre si
margen_real > spread_line. Las lineas de la casa llegan en formato casa
(local -2.5 = local favorito por 2.5) y aqui se convierten.
"""
import json
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "history.json"

# ventana inicial (+- puntos de linea), muestra efectiva minima y ventana maxima
VENTANA_SPREAD = 1.0
VENTANA_TOTAL = 1.5
N_EFECTIVO_MIN = 200
VENTANA_MAX = 4.0

_HIST = None


def cargar_historico(path: Path = HISTORY_PATH):
    """Lista de filas (season, week, spread_line, total_line, margin, total)."""
    global _HIST
    if _HIST is None or path != HISTORY_PATH:
        data = json.loads(Path(path).read_text())
        rows = [tuple(r) for r in data["rows"]]
        if path != HISTORY_PATH:
            return rows
        _HIST = rows
    return _HIST


def distribucion_condicionada(historico, valor, col_linea: int, col_resultado: int,
                              ventana: float, n_min: int = N_EFECTIVO_MIN):
    """pmf empirica {resultado_entero: prob} de los partidos con linea cercana a `valor`.

    Pesos triangulares (1 en la linea exacta, 0 a `ventana` de distancia). Si la
    muestra efectiva es menor a `n_min`, la ventana crece de 0.5 en 0.5 hasta
    VENTANA_MAX. Devuelve (pmf, n_efectivo, ventana_usada).
    """
    h = ventana
    while True:
        pesos = {}
        s1 = s2 = 0.0
        for r in historico:
            d = abs(r[col_linea] - valor)
            if d < h:
                w = 1.0 - d / h
                k = int(r[col_resultado])
                pesos[k] = pesos.get(k, 0.0) + w
                s1 += w
                s2 += w * w
        n_ef = (s1 * s1 / s2) if s2 > 0 else 0.0
        if n_ef >= n_min or h >= VENTANA_MAX:
            break
        h += 0.5
    if s1 <= 0:
        return {}, 0.0, h
    return {k: v / s1 for k, v in pesos.items()}, n_ef, h


def equilibrar(pmf, v):
    """Ajusta la pmf para que, en la linea del mercado `v`, cubrir / no cubrir sea
    50/50 (descontando el empate exacto). El mercado de cierre es eficiente: la
    ventana de partidos con linea cercana mezcla lineas distintas y sesga el centro,
    pero conserva bien la FORMA (picos en 3, 7, 6, 10, 14 y la masa de empate). Se
    reescala cada lado por separado y se deja intacta la masa en la linea."""
    p_eq = pmf.get(int(v), 0.0) if float(v).is_integer() else 0.0
    arriba = sum(p for k, p in pmf.items() if k > v)
    abajo = sum(p for k, p in pmf.items() if k < v)
    if arriba <= 0 or abajo <= 0:
        return dict(pmf)
    meta = (1.0 - p_eq) / 2
    return {k: (p * meta / arriba if k > v else (p * meta / abajo if k < v else p))
            for k, p in pmf.items()}


def prob_mayor(pmf, x):
    return sum(p for k, p in pmf.items() if k > x)


def prob_menor(pmf, x):
    return sum(p for k, p in pmf.items() if k < x)


def prob_igual(pmf, x):
    return pmf.get(int(x), 0.0) if float(x).is_integer() else 0.0


def moneyline_a_prob(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def momio_justo(p):
    """Momio americano sin margen para la probabilidad p (0-1)."""
    if p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def ev_100(p_gana, p_push, o, stake=100.0):
    """VE por $100: gana (p_gana) cobra segun el momio, push devuelve, resto pierde."""
    ganancia = stake * (o / 100) if o > 0 else stake * (100 / -o)
    return round(p_gana * ganancia - (1.0 - p_gana - p_push) * stake, 2)


def comparar_casa(spread_consenso, total_consenso, cuotas: dict = None,
                  spread_casa_local=None, total_casa=None, historico=None):
    """Compara tu casa contra el consenso y calcula el VE por $100 de cada apuesta.

    spread_consenso / total_consenso: linea nflverse (spread_line > 0 = local favorito).
    cuotas: momios americanos de tu casa: sp_home, sp_away, over, under, ml_home, ml_away.
    spread_casa_local: linea del LOCAL en tu casa, formato casa (-2.5 = favorito por 2.5).
        Si falta, se asume la del consenso.
    total_casa: total de tu casa; si falta, el del consenso.
    Devuelve None si no hay cuotas que evaluar.
    """
    cuotas = {k: v for k, v in (cuotas or {}).items() if v is not None}
    if not cuotas:
        return None
    hist = historico if historico is not None else cargar_historico()

    x_spread = spread_consenso if spread_casa_local is None else -spread_casa_local  # el local cubre si margen > x
    x_total = total_consenso if total_casa is None else total_casa

    pmf_m, n_m, h_m = distribucion_condicionada(hist, spread_consenso, 2, 4, VENTANA_SPREAD)
    pmf_t, n_t, h_t = distribucion_condicionada(hist, total_consenso, 3, 5, VENTANA_TOTAL)
    # el consenso es la verdad: en SU linea, cubrir y no cubrir valen lo mismo
    pmf_m = equilibrar(pmf_m, spread_consenso)
    pmf_t = equilibrar(pmf_t, total_consenso)

    filas = []

    def fila(clave, etiqueta, tipo, o, p_gana, p_push, n_ef, linea_txt):
        p_efect = p_gana / (1 - p_push) if p_push < 1 else p_gana
        filas.append({
            "clave": clave, "etiqueta": etiqueta, "tipo": tipo, "linea": linea_txt,
            "cuota": o, "p_gana": p_gana * 100, "p_push": p_push * 100,
            "cuota_justa": momio_justo(p_efect), "ev": ev_100(p_gana, p_push, o), "n_ef": n_ef,
        })

    push_s = prob_igual(pmf_m, x_spread)
    if "sp_home" in cuotas:
        fila("sp_home", "{home}", "spread", cuotas["sp_home"], prob_mayor(pmf_m, x_spread), push_s, n_m,
             f"{-x_spread:+.1f}")
    if "sp_away" in cuotas:
        fila("sp_away", "{away}", "spread", cuotas["sp_away"], prob_menor(pmf_m, x_spread), push_s, n_m,
             f"{x_spread:+.1f}")
    push_t = prob_igual(pmf_t, x_total)
    if "over" in cuotas:
        fila("over", "Over", "total", cuotas["over"], prob_mayor(pmf_t, x_total), push_t, n_t, f"{x_total:g}")
    if "under" in cuotas:
        fila("under", "Under", "total", cuotas["under"], prob_menor(pmf_t, x_total), push_t, n_t, f"{x_total:g}")
    # ganador: el empate (0) devuelve el dinero, igual que en el spread
    push_ml = prob_igual(pmf_m, 0)
    if "ml_home" in cuotas:
        fila("ml_home", "{home} ML", "ganador", cuotas["ml_home"], prob_mayor(pmf_m, 0), push_ml, n_m, "")
    if "ml_away" in cuotas:
        fila("ml_away", "{away} ML", "ganador", cuotas["ml_away"], prob_menor(pmf_m, 0), push_ml, n_m, "")

    def overround(a, b):
        if a in cuotas and b in cuotas:
            return round((moneyline_a_prob(cuotas[a]) + moneyline_a_prob(cuotas[b]) - 1) * 100, 1)
        return None

    return {
        "consenso_spread": spread_consenso, "consenso_total": total_consenso,
        "casa_x_spread": x_spread, "casa_total": x_total,
        "casa_distinta_spread": spread_casa_local is not None and x_spread != spread_consenso,
        "casa_distinta_total": total_casa is not None and x_total != total_consenso,
        "n_ef_spread": round(n_m), "ventana_spread": h_m, "n_ef_total": round(n_t), "ventana_total": h_t,
        "push_spread": push_s * 100, "push_total": push_t * 100,
        "margen_casa": {"spread": overround("sp_home", "sp_away"), "total": overround("over", "under"),
                        "ganador": overround("ml_home", "ml_away")},
        "filas": filas,
    }
