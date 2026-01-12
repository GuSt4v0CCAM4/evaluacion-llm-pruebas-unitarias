# An Initial Investigation of ChatGPT Unit Test Generation Capability

Este repositorio contiene los artefactos y código del estudio **"An initial investigation of ChatGPT unit test generation capability"** publicado en SAST'2023.

## 📋 Resumen

Este proyecto evalúa la capacidad de ChatGPT (GPT-3.5-turbo) para generar pruebas unitarias automáticas en Java, comparándolo con herramientas tradicionales de generación de tests como EvoSuite.

## 🎯 Objetivos

1. Analizar el efecto del parámetro `temperature` en la calidad de los tests generados por ChatGPT
2. Comparar ChatGPT con herramientas tradicionales de generación automática de tests (EvoSuite)
3. Evaluar la efectividad usando métricas de cobertura de código y mutation score

## 📊 Metodología

### Conjunto de Datos
- **33 proyectos Java** implementando algoritmos y estructuras de datos clásicas
- Categorías: búsqueda de máximos/mínimos, ordenación, Fibonacci, listas, pilas, filas, tablas hash, árboles binarios, grafos, y algoritmos de casamiento de patrones

### Generación de Tests
Para cada proyecto:
- **11 configuraciones de temperature** (0.0 a 1.0 con paso de 0.1)
- **3 réplicas** por configuración
- **Total:** 33 tests por proyecto (11 × 3)
- **Prompt:** "Generate test cases just for the [ClassName] Java class in one java class file with imports using JUnit 4 and Java 8"

### Métricas Evaluadas
1. **Cobertura de Código:** Porcentaje de líneas ejecutadas por los tests
2. **Mutation Score:** Porcentaje de mutantes detectados (mide efectividad en encontrar bugs)
3. **Tasa de Éxito:** Cantidad de tests que compilan y ejecutan correctamente

## 📈 Resultados Principales

### Efecto de Temperature
| Temperature | Tests Válidos | Cobertura | Mutation Score |
|-------------|---------------|-----------|----------------|
| 0.0         | 69            | 46.57%    | 32.25%         |
| 0.5         | 35            | **90.40%**| 67.26%         |
| 0.6         | **52**        | 87.40%    | 68.21%         |
| 0.8         | 45            | 89.78%    | **68.53%**     |

**Hallazgos:**
- Temperature 0.0 genera muchos tests defectuosos (compilación fallida)
- Temperaturas entre 0.5-0.8 producen los mejores resultados
- Temperature 0.5: Mayor cobertura (90.4%)
- Temperature 0.8: Mejor mutation score (68.5%)

### ChatGPT vs EvoSuite
En los 8 proyectos comparados, ChatGPT superó a EvoSuite en todos los casos:
- **Diferencia promedio:** +5.4 puntos porcentuales en mutation score
- **Rango de ventaja:** +2.6 a +11.6 puntos

### Resultados por Tipo de Algoritmo
- **Algoritmos simples** (Fibonacci, Max/Min): Excelentes (100% mutation score en Fibonacci)
- **Estructuras de datos** (Lista, Pilha): Muy buena cobertura (>97%), mutation score moderado (71-81%)
- **Algoritmos complejos** (CasamentoExato): Alta cobertura (>95%), pero bajo mutation score (28-32%)

## 🚀 Implementación

### Requisitos Previos
```bash
# Requisitos del sistema
- Java 8 o superior
- Maven 3.x
- Python 3.8 o superior
- pip (gestor de paquetes Python)
```

### Instalación

#### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd chatgpt
```

#### 2. Instalar Dependencias Python
```bash
# Crear entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

#### 3. Verificar Instalación de Java y Maven
```bash
java -version  # Debe ser Java 8+
mvn -version   # Debe ser Maven 3.x
```

### Estructura del Proyecto
```
.
├── projetos/              # 33 proyectos Java evaluados
│   └── 01Max/
│       ├── src/           # Código fuente
│       │   ├── main/java/ds/      # Clase bajo prueba
│       │   └── test/java/ds/      # Tests (vacío inicialmente)
│       ├── gpt-tests/     # Tests pre-generados por ChatGPT
│       ├── reports/       # Reportes de Pitest (generados al ejecutar)
│       └── pom.xml        # Configuración Maven con Pitest
├── scripts/               # Scripts de generación y evaluación
│   ├── gera-chatgpt.py   # Generador de tests con ChatGPT
│   ├── reproduce_evaluation.py  # Script de reproducción
│   ├── reports-chatgpt.py # Extrae métricas de reportes HTML
│   ├── aggregate_reports.py     # Agrega datos en CSVs
│   ├── analyze_metrics_and_anova.py  # Análisis estadístico
│   └── files.txt          # Lista de proyectos a evaluar
├── generated_reports/     # Resultados pre-calculados
│   ├── all.csv            # Datos brutos
│   ├── metrics_by_temperature.csv
│   └── metrics_by_project_LLM.csv
├── requirements.txt       # Dependencias Python
└── README.md              # Este archivo
```

### Reproducir la Evaluación (Sin API de OpenAI)

Los tests ya están pre-generados en cada proyecto (`gpt-tests/`). Para ejecutar la evaluación completa:

```bash
cd scripts
python3 reproduce_evaluation.py
```

Este script:
1. Lee la lista de proyectos de `files.txt`
2. Para cada proyecto y cada test (0-33):
   - Copia el test desde `gpt-tests/` a `src/test/java/ds/`
   - Ejecuta `mvn clean install` y Pitest para mutation testing
   - Registra si el test pasa o falla
3. Genera un resumen JSON con estadísticas

**Nota:** El proceso completo puede tardar **varias horas** (33 proyectos × 34 tests cada uno).

**Tip:** Para probar con un solo proyecto, edita `files.txt` temporalmente.

### Generar Nuevos Tests (Requiere API de OpenAI)

Si deseas generar nuevos tests desde cero:

#### 1. Obtener API Key de OpenAI
Regístrate y obtén tu API key en https://platform.openai.com/api-keys

#### 2. Configurar la API Key
Edita `scripts/gera-chatgpt.py` y busca la línea 23:
```python
"Authorization": "Bearer YOUR_OPENAI_API_KEY_HERE"  # Reemplaza con tu API key
```

Reemplaza `YOUR_OPENAI_API_KEY_HERE` con tu key real.

#### 3. Ejecutar el Generador
```bash
cd scripts
python3 gera-chatgpt.py
```

**Costos:** Ten en cuenta que esto hará ~1,000 llamadas a la API de OpenAI (33 proyectos × ~33 tests).

### Analizar Resultados

Después de ejecutar `reproduce_evaluation.py`, puedes analizar los resultados:

```bash
cd scripts

# Generar reportes agregados
python3 reports-chatgpt.py  # Extrae métricas de los reportes HTML de Pitest
python3 aggregate_reports.py  # Agrega en CSVs por temperatura y proyecto

# Análisis estadístico (ANOVA)
python3 analyze_metrics_and_anova.py
```

## 📁 Datos y Resultados

Todos los datos están disponibles en `generated_reports/`:

- **`all.csv`**: Datos completos de 474 ejecuciones exitosas
- **`metrics_by_temperature.csv`**: Agregación por temperatura
- **`metrics_by_project_LLM.csv`**: Métricas de ChatGPT por proyecto
- **`metrics_by_project.csv`**: Comparación con herramientas tradicionales

## 🔍 Proyectos Evaluados

| ID | Nombre | Tipo | Complejidad |
|----|--------|------|-------------|
| 01-04 | Max/MaxMin | Búsqueda | Baja |
| 05, 09, 17 | Ordenacao | Ordenación | Media |
| 06-07 | Fibonacci | Recursión | Baja |
| 11-12 | Lista | Estructura de datos | Media |
| 13-14 | Pilha | Estructura de datos | Media |
| 15-16 | Fila | Estructura de datos | Media |
| 20-23 | Tabela/TabelaHash | Hashing | Alta |
| 21 | ArvoreBinaria | Árboles | Alta |
| 24-29 | Grafo | Grafos | Alta |
| 31-32 | Casamento | Pattern matching | Muy Alta |
| 33 | Identifier | Validación | Media |


**Nota:** Los resultados pueden variar al regenerar tests debido a la naturaleza no determinista de los modelos de lenguaje, incluso con temperature fijo.
