# Contributing to ChatGPT Unit Test Generation Evaluation

¡Gracias por tu interés en contribuir a este proyecto! Este documento proporciona guías para contribuir efectivamente.

## 📋 Tabla de Contenidos

- [Configuración del Entorno](#configuración-del-entorno)
- [Estándares de Código](#estándares-de-código)
- [Agregar Nuevos Proyectos](#agregar-nuevos-proyectos)
- [Ejecutar Tests](#ejecutar-tests)
- [Reportar Issues](#reportar-issues)
- [Pull Requests](#pull-requests)

## 🛠️ Configuración del Entorno

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd chatgpt
```

### 2. Configurar Entorno Python
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar herramientas de desarrollo (opcional)
pip install black flake8 pylint
```

### 3. Verificar Configuración
```bash
# Verificar Java y Maven
java -version  # Debe ser Java 8+
mvn -version   # Maven 3.x

# Verificar Python
python3 --version  # Python 3.8+
```

## 📝 Estándares de Código

### Python

Este proyecto sigue [PEP 8](https://peps.python.org/pep-0008/) para código Python.

#### Guías Principales:
- **Indentación:** 4 espacios (no tabs)
- **Longitud de línea:** Máximo 100 caracteres
- **Imports:** Agrupados en stdlib, third-party, local
- **Docstrings:** Formato Google Style para todas las funciones públicas

#### Ejemplo de Docstring:
```python
def process_report(file_path, temperature):
    """Procesa un reporte de Pitest y extrae métricas.
    
    Args:
        file_path: Ruta absoluta al archivo HTML del reporte
        temperature: Valor de temperatura usado (0.0-1.0)
    
    Returns:
        dict: Diccionario con métricas de cobertura y mutation score
    
    Raises:
        FileNotFoundError: Si el archivo no existe
    """
```

#### Herramientas Recomendadas:
```bash
# Formatear código automáticamente
black scripts/*.py

# Verificar estilo
flake8 scripts/*.py --max-line-length=100

# Análisis estático
pylint scripts/*.py
```

### Java

- Seguir convenciones estándar de Java
- Usar JUnit 4 para tests
- Mantener compatibilidad con Java 8

## ➕ Agregar Nuevos Proyectos

Para agregar un nuevo proyecto Java a la evaluación:

### 1. Estructura del Proyecto
Crear la siguiente estructura en `projetos/`:

```
projetos/
└── XX_NombreProyecto/
    ├── src/
    │   └── main/java/ds/
    │       └── ClaseAPrueba.java
    ├── gpt-tests/
    │   └── (tests pre-generados)
    └── pom.xml
```

### 2. Configurar pom.xml
Copiar y adaptar `pom.xml` de un proyecto existente. Asegurarse de:
- Incluir dependencias de JUnit 4
- Configurar Pitest plugin
- Especificar la clase bajo prueba en `<targetClasses>`

### 3. Actualizar files.txt
Agregar una línea en `scripts/files.txt`:
```
XX_NombreProyecto:ds.NombreClase
```

### 4. Generar Tests
```bash
cd scripts
python3 gera-chatgpt.py  # Requiere API key configurada
```

## 🧪 Ejecutar Tests

### Evaluar un Solo Proyecto
```bash
cd projetos/01Max
mvn clean test
```

### Ejecutar Mutation Testing
```bash
cd projetos/01Max
mvn org.pitest:pitest-maven:mutationCoverage
```

El reporte HTML se generará en `target/pit-reports/`.

### Reproducir Evaluación Completa
```bash
cd scripts
python3 reproduce_evaluation.py
```

## 🐛 Reportar Issues

Al reportar un issue, incluye:

1. **Descripción clara** del problema
2. **Pasos para reproducir**
3. **Comportamiento esperado vs. actual**
4. **Versiones:**
   - Python: `python3 --version`
   - Java: `java -version`
   - Maven: `mvn -version`
   - OS: `uname -a` o Windows version
5. **Logs relevantes** (si aplica)

### Ejemplo de Issue:
```
## Descripción
El script reproduce_evaluation.py falla al procesar el proyecto 05_Ordenacao

## Pasos para Reproducir
1. cd scripts
2. python3 reproduce_evaluation.py
3. Error ocurre en proyecto 05_Ordenacao, test 12

## Error
FileNotFoundError: .../05_Ordenacao/reports/OrdenacaoTest12/index.html

## Versiones
- Python: 3.10.12
- Java: openjdk 11.0.20
- Maven: 3.8.7
- Ubuntu 22.04
```

## 🔄 Pull Requests

### Antes de Crear un PR:

1. **Sincronizar con main:**
   ```bash
   git checkout main
   git pull origin main
   git checkout tu-branch
   git rebase main
   ```

2. **Verificar código:**
   ```bash
   # Formatear
   black scripts/*.py
   
   # Verificar estilo
   flake8 scripts/*.py --max-line-length=100
   
   # Probar sintaxis
   python3 -m py_compile scripts/*.py
   ```

3. **Commits descriptivos:**
   ```bash
   git commit -m "feat: Agregar análisis de cobertura por tipo de test"
   git commit -m "fix: Corregir parsing de reportes con múltiples clases"
   git commit -m "docs: Actualizar README con instrucciones de instalación"
   ```

### Formato del PR:

**Título:** Breve descripción (ej: "Agregar soporte para Java 11")

**Descripción debe incluir:**
- ¿Qué cambia este PR?
- ¿Por qué es necesario?
- ¿Cómo se probó?
- Referencias a issues relacionados (si aplica)

### Ejemplo:
```markdown
## Cambios
- Actualiza pom.xml para soportar Java 11
- Modifica scripts para detectar versión de Java automáticamente

## Motivación
Muchos desarrolladores usan Java 11+, pero el proyecto estaba limitado a Java 8.

## Pruebas
- ✅ Ejecutado en Java 8, 11, y 17
- ✅ Todos los tests pasan
- ✅ Pitest genera reportes correctamente

## Issues Relacionados
Closes #42
```

## 💡 Consejos Adicionales

- **Comunicación:** Antes de trabajar en features grandes, abre un issue para discutir
- **Incrementalidad:** Preferible hacer PRs pequeños y frecuentes que grandes cambios
- **Documentación:** Actualiza README.md y docstrings cuando cambies funcionalidad
- **Testing:** Si agregas código Python, considera agregar ejemplos de uso en docstrings

## 📞 Contacto

Para preguntas o discusiones, abre un issue en GitHub.

---

¡Gracias por contribuir! 🎉
