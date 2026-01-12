import pandas as pd
import numpy as np
import os
from pathlib import Path

# Requerido para ANOVA/Tukey
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

IN = Path("scripts/reports/all.csv")
OUTDIR = Path("generated_reports")
OUTDIR.mkdir(exist_ok=True, parents=True)

if not IN.exists():
    raise SystemExit(f"File not found: {IN} (ej : ejecuta scripts/reports-chatgpt.py o genera los reports)")

# Leer CSV robusto
df = pd.read_csv(IN, sep=';', dtype=str).fillna('')

# Normalizar columnas: permitir que vengan como '75%' o '75' o ya numéricas
def pct_to_float(s):
    """Convierte string con porcentaje a float.
    
    Args:
        s: String con formato '75%' o '75'
    
    Returns:
        float or nan: Valor numérico o np.nan si no se puede convertir
    """
    if s is None: return np.nan
    s = str(s).strip()
    if s == '': return np.nan
    # extraer primer número (con . o ,)
    import re
    m = re.search(r'(-?\d+(?:[.,]\d+)?)', s)
    if not m:
        return np.nan
    val = m.group(1).replace(',', '.')
    try:
        return float(val)
    except:
        return np.nan

# columnas limpias
df['Cob_num'] = df['Cobertura'].apply(pct_to_float)
df['Esc_num'] = df['Escore'].apply(pct_to_float)

# Gerados (número de tests generados en ese registro)
# intenta convertir directamente; si es string con no num, extrae dígitos
def parse_int_like(s):
    """Extrae valor entero de una cadena.
    
    Args:
        s: String que puede contener dígitos
    
    Returns:
        int or nan: Valor entero o np.nan si no se encuentra
    """
    if pd.isna(s): return np.nan
    s = str(s).strip()
    import re
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else np.nan

df['Gerados_num'] = pd.to_numeric(df['Gerados'].apply(lambda x: parse_int_like(x)), errors='coerce')

# Temperature num
df['Temp_num'] = df['Temperature'].astype(str).str.replace(',', '.').apply(lambda x: float(x) if x!='' else np.nan)

# Agregación por temperatura
grp_temp = df.groupby('Temp_num').agg(
    n_success = ('PRJ','count'),
    cov_mean = ('Cob_num','mean'),
    cov_std = ('Cob_num','std'),
    esc_mean = ('Esc_num','mean'),
    esc_std = ('Esc_num','std'),
    ger_med = ('Gerados_num','median')
).reset_index().sort_values('Temp_num')

# success_pct sobre 33*3 = 99 (si ese es el diseño)
TOTAL = 33 * 3
grp_temp['success_pct'] = (grp_temp['n_success'] / TOTAL) * 100

# Guardar CSVs
grp_temp.to_csv(OUTDIR / "metrics_by_temperature_clean.csv", index=False, float_format="%.4f", sep=';')

# Agregación por proyecto (promedio sobre temperaturas/réplicas)
grp_proj = df.groupby('PRJ').agg(
    N_records = ('Temp_num','count'),
    Coverage_mean = ('Cob_num','mean'),
    Coverage_std = ('Cob_num','std'),
    Escore_mean = ('Esc_num','mean'),
    Escore_std = ('Esc_num','std')
).reset_index().sort_values('PRJ')
grp_proj.to_csv(OUTDIR / "metrics_by_project_LLM_generated.csv", index=False, sep=';')

print("Wrote:", OUTDIR / "metrics_by_temperature_clean.csv")
print("Wrote:", OUTDIR / "metrics_by_project_LLM_generated.csv")
print("\nSummary by temperature:")
print(grp_temp.to_string(index=False, float_format='%.2f'))

# ----------------------
# ANOVA sobre Escore_num ~ Temp
# excluye nulos
df_anova = df.dropna(subset=['Esc_num','Temp_num']).copy()
if len(df_anova) < 2:
    print("No hay suficientes datos para ANOVA.")
else:
    model = ols('Esc_num ~ C(Temp_num)', data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nANOVA result (Esc_num ~ Temp_num):")
    print(anova_table)

    # Tukey HSD post-hoc
    try:
        tuk = pairwise_tukeyhsd(df_anova['Esc_num'], df_anova['Temp_num'].astype(str), alpha=0.05)
        print("\nTukey HSD summary:")
        print(tuk.summary())
    except Exception as e:
        print("Tukey HSD failed:", e)
