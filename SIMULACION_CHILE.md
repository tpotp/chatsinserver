# Simulacion Chile

## Lanzar el host con colmena simulada

Desde esta carpeta:

```bat
launch_colmena_chile.bat --simulate 5
```

O con Python:

```bash
python launch_colmena_chile.py --simulate 5
```

El script:

- genera una sala nueva,
- abre el host en `https://chatsinserver.vercel.app/index.html?...&simulate=N`,
- copia al portapapeles la URL del celular,
- imprime tambien una URL `assistOnly` para probar desde otro navegador sin shards locales.

## Flujo recomendado para la demo

1. Lanza el host en tu PC.
2. Espera a que el host muestre `Cerebro local ... listo` y que el Pudú pase a `La colmena se ha creado`.
3. Abre la URL `Celular` desde tu Redmi 10C en Entel 4G o 5G.
4. Si quieres probar desde otro navegador del mismo PC, usa la URL `Cliente liviano`.

## Notas

- `simulate` acepta de `2` a `12`.
- `assistOnly=1` fuerza un cliente liviano: no intenta cargar shards locales y enruta a la colmena.
- En simulación, la colmena usa `shards virtuales` para activarse rápido y el host carga un cerebro local colectivo para responder mejor.
- El Pudú ahora tiene panel movil, tacto, susurros, voz opcional y movimiento por sensor.
