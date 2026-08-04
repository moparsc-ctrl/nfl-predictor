# NFL Predictor

Modelo de prediccion de resultados NFL (Poisson + EPA), validado con la
temporada 2025: ~64.6% de acierto en ganador, ~10.5 pts de error promedio
en el total combinado.

## Estructura

- `model/` — logica del modelo (ratings + Poisson), reusable.
- `scripts/update_cache.py` — descarga datos de nflverse y genera
  `data/stats_cache.json`. Se corre solo (via GitHub Actions, semanal).
- `.github/workflows/update-cache.yml` — cron que corre el update cada
  lunes y commitea el cache actualizado al repo.
- `api/predict.py` — funcion serverless de Vercel. Lee el cache (no
  recalcula nada pesado) y devuelve una ficha JPG.

## Uso

**Interfaz web:** abre la raiz del proyecto (`/`) — hay una pagina con
dos selectores (local / visitante) y un boton que genera la ficha.

**Directo por URL:**
```
GET /api/predict?home=KC&away=BUF
```

Devuelve una imagen JPEG con forma reciente, probabilidades, marcador
esperado, total combinado (over/under) y recomendacion.

## Deploy

1. Sube este repo a GitHub.
2. Corre una vez a mano `python scripts/update_cache.py` para generar
   el primer `data/stats_cache.json` (o dispara el workflow manualmente
   desde la pestaña Actions -> "Run workflow").
3. Importa el repo en Vercel — detecta `vercel.json` automaticamente.
4. Cada lunes el cache se actualiza solo; Vercel sirve siempre datos
   frescos sin recalcular nada en cada request.

## Siguientes mejoras posibles

- Afinar la logica de `recomendacion_apuesta` (umbral de favorito,
  doble oportunidad 1X/X2, etc.)
- Agregar splits home/away separados en vez de promedio general.
- Comparar contra `spread_line` / `total_line` reales del dataset para
  medir edge contra el mercado.
