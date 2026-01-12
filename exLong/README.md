# Replicación del Paper exLong

## Generating Exceptional Behavior Tests with Large Language Models

Este directorio contiene la infraestructura para replicar los resultados del proyecto **exLong**, un modelo basado en LLMs para la generación de pruebas de comportamiento excepcional en Java.

---

## 🔗 Repositorio Oficial
El código fuente original y el paquete de replicación oficial se encuentran en:
**[EngineeringSoftware/exLong](https://github.com/EngineeringSoftware/exLong)**

---

## 📊 Resumen de la Replicación

Esta infraestructura permite validar los hallazgos principales del paper mediante el procesamiento de métricas de similitud y ejecución.

### Casos de Uso Evaluados
- **RQ1 (Developer-oriented)**: Evaluación orientada al desarrollador utilizando métricas como BLEU, CodeBLEU y tasas de compilación.
- **RQ2 (Machine-oriented)**: Evaluación orientada a la máquina analizando la cobertura de sentencias `throw` y la efectividad automática.

---

## 🛠️ Scripts de Replicación

1. **`ejecutar_replicacion.py`**: El punto de entrada principal. Ejecuta secuencialmente los análisis y genera los gráficos.
2. **`replicacion_rq1.py`**: Procesa los resultados del caso de uso orientado al desarrollador.
3. **`replicacion_rq2.py`**: Procesa los resultados del caso de uso orientado a la máquina.
4. **`generar_graficos.py`**: Genera visualizaciones SVG profesionales en la carpeta `resultados_replicacion/`.

---

## 📈 Cómo Ejecutar

Para ejecutar la replicación completa y generar los gráficos:

```bash
python3 ejecutar_replicacion.py
```

Los resultados se guardarán en: `resultados_replicacion/`

---

## 📁 Estructura del Proyecto

```
exLong/
├── ejecutar_replicacion.py   # Script principal
├── replicacion_rq1.py        # Análisis RQ1
├── replicacion_rq2.py        # Análisis RQ2
├── generar_graficos.py       # Generador de gráficos SVG
├── exLong/                   # Clon del repositorio oficial
└── resultados_replicacion/   # 📊 Resultados y gráficos generados
```

---

## 📝 Notas
Los scripts están diseñados para ser robustos y no requieren dependencias externas (utilizan la biblioteca estándar de Python y generan SVGs directamente). Detectarán automáticamente los resultados reales si se encuentran en las carpetas de salida estándar del proyecto `exLong`.
