"""Parametros configurables del modelo. Todos se pueden cambiar aqui o pasarse
como argumento a las funciones; ninguno esta fijo dentro de la logica.

Valores respaldados por backtest walk-forward 2018-2025 (ver README / resultados):
"""

# --- Probabilidad de victoria -------------------------------------------------
# "normal": P(local) = Phi(margen_esperado / sigma)   (calibrado)
# "poisson": suma exacta sobre Poisson (sobreconfiado: sigma implicita ~6-7 pts)
METODO_PROB = "normal"

# Desviacion de (margen real - margen esperado del modelo): 13.63 en el backtest.
SIGMA_MARGEN_MODELO = 13.6

# Desviacion de (total real - total esperado): 13.88 en el backtest.
SIGMA_TOTAL = 13.9

# Correccion al total esperado del modelo (se reparte mitad y mitad entre ambos
# equipos, asi el marcador mostrado sigue sumando el total). En el backtest
# (real - modelo) = -0.97 => el modelo se pasa ~1 pt: la correccion correcta es -1.0.
# Ojo: el sesgo varia mucho por temporada (+0.6 a -5.0), asi que es un ajuste debil.
TOTAL_BIAS = -1.0

# --- Mercado ------------------------------------------------------------------
# Sigma para convertir spread_line en probabilidad (mejor ajuste historico 12.5-13.9).
SIGMA_MARGEN_MERCADO = 13.86

# Peso del mercado en la mezcla. En el backtest el Brier/log loss mejora al subir
# el peso del mercado y el minimo esta en 1.0 (el modelo no aporta informacion
# que el mercado no tenga). Es configurable: 0.65 era el valor anterior.
MARKET_WEIGHT = 1.0

# --- Modelo de puntos ---------------------------------------------------------
# El backtest 2018-2025 muestra que el ajuste EPA (peso 0.08) EMPEORA el modelo
# (Brier 0.2439 con EPA contra 0.2322 sin EPA) y que no aporta informacion que el
# mercado no tenga. Por eso queda apagado; ponlo en True para usar los ratings EPA.
USAR_EPA = False

# --- Numeros clave (3, 6, 7, 10, 14) para probabilidad de cubrir spread --------
# Backtest fuera de muestra: no mejora la calibracion, por eso esta desactivado.
NUMEROS_CLAVE = False
# Razon frecuencia real / normal, estimada con 2015-2020 (indicativa).
NUMEROS_CLAVE_BOOST = {3: 2.68, 6: 1.38, 7: 1.92, 10: 1.13, 14: 1.70}

# --- Recomendacion ------------------------------------------------------------
# Edge (pts) a partir del cual se muestra la diferencia modelo vs mercado.
UMBRAL_VALOR = 3.0
EDGE_ALTO = 15.0
# Umbral de edge con ROI positivo validado en backtest. None = ninguno lo fue
# (ver resultados de rentabilidad): la ficha dice "Diferencia ... (no validada)".
EDGE_VALIDADO = None
