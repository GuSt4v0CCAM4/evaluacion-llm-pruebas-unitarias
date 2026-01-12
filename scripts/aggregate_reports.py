#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import re

reports_dir = Path("scripts/reports")
out_dir = Path("generated_reports")
out_dir.mkdir(exist_ok=True)

# leer todos los CSV (excepto all.csv si ya existe)
dfs = []
for csv_file in sorted(reports_dir.glob("*.csv")):
    if csv_file.name == "all.csv":
        continue
    try:
        df = pd.read_csv(csv_file, sep=";", dtype=str, encoding="utf-8", on_bad_lines="skip")
        df["__source_file"] = csv_file.name
        dfs.append(df)
    except Exception as e:
        print(f"[warn] no pude leer {csv_file}: {e}")

if not dfs:
    raise SystemExit("[error] No se encontraron CSVs o no se pudieron leer.")

# concatenar
df_all = pd.concat(dfs, ignore_index=True, sort=False)

# Normalizar nombres de columnas (por si hay espacios u otros)
df_all.columns = [c.strip() for c in df_all.columns]

# --- Limpieza de columnas numéricas ---
def parse_percent_column(s):
    """Extrae el primer porcentaje numérico de una cadena y lo devuelve como float (0-100).
    
    Args:
        s: String que puede contener porcentajes (ej: '75%', '0%0%98%')
    
    Returns:
        float or None: Valor numérico del porcentaje, o None si no se encuentra
    """
    if pd.isna(s):
        return None
    # Si la celda contiene varios porcentajes concatenados, buscamos el primero plausible
    # Ej: '0%0%0%98%' -> encontrará '0' como primer match; preferimos el último significativo, así que buscamos todos.
    percents = re.findall(r'(-?\d+(?:[.,]\d+)?)\s*%', str(s))
    if percents:
        # elegir el último valor numérico mayor que 0 si existe, sino el último encontrado
        nums = []
        for p in percents:
            tmp = p.replace(",", ".")
            try:
                nums.append(float(tmp))
            except:
                pass
        if nums:
            # heurística: si hay varios, tomar la media o el máximo; usamos el promedio de valores numéricos distintos
            return float(sum(nums) / len(nums))
    # fallback: intentar extraer cualquier número en la cadena
    numbers = re.findall(r'(-?\d+(?:[.,]\d+)?)', str(s))
    if numbers:
        try:
            return float(numbers[-1].replace(",", "."))
        except:
            return None
    return None

def parse_int_like(s):
    """Extrae valor entero de una cadena.
    
    Args:
        s: String que puede contener un número
    
    Returns:
        int or None: Valor entero extraído, o None si no se encuentra
    """
    if pd.isna(s):
        return None
    s2 = re.sub(r'[^\d\-]', '', str(s))
    if s2 == '':
        return None
    try:
        return int(s2)
    except:
        try:
            return int(float(s2))
        except:
            return None

# Crear/convertir columnas esperadas si existen
# Columnas originales: PRJ;Temperature;Cobertos;Gerados;Cobertura;Mortos;Total;Escore
if "Cobertura" in df_all.columns:
    df_all["Cobertura_num"] = df_all["Cobertura"].apply(parse_percent_column)
else:
    df_all["Cobertura_num"] = pd.NA

for col in ["Gerados", "Cobertos", "Mortos", "Total"]:
    if col in df_all.columns:
        df_all[f"{col}_num"] = df_all[col].apply(parse_int_like)
    else:
        df_all[f"{col}_num"] = pd.NA

# Escore: puede estar en % o ser número
if "Escore" in df_all.columns:
    df_all["Escore_num"] = df_all["Escore"].apply(parse_percent_column)
else:
    df_all["Escore_num"] = pd.NA

# Temperature -> float (int/float con coma o punto)
if "Temperature" in df_all.columns:
    def parse_temp(x):
        if pd.isna(x): return None
        s = str(x).strip()
        s = s.replace(",", ".")
        # extraer primer número
        m = re.search(r'-?\d+(\.\d+)?', s)
        return float(m.group(0)) if m else None
    df_all["Temperature_num"] = df_all["Temperature"].apply(parse_temp)
else:
    df_all["Temperature_num"] = pd.NA

# Guardar all.csv "limpio" (separador ;)
all_csv_path = out_dir / "all.csv"
df_all.to_csv(all_csv_path, sep=";", index=False)
print(f"[ok] all.csv generado en {all_csv_path} (raw concatenated)")

# Agrupar por temperatura numérica usando columnas limpias
group_key = "Temperature_num"
agg = df_all.groupby(group_key).agg(
    Cobertura_media=("Cobertura_num", "mean"),
    Escore_media=("Escore_num", "mean"),
    Gerados_medios=("Gerados_num", "mean"),
    Mortos_medios=("Mortos_num", "mean"),
    Total_mutantes=("Total_num", "mean"),
    N_registros=("PRJ", "count")
).reset_index().sort_values(group_key)

by_temp_path = out_dir / "metrics_by_temperature.csv"
agg.to_csv(by_temp_path, sep=";", index=False)
print(f"[ok] metrics_by_temperature.csv generado en {by_temp_path}")

# Métricas por proyecto (usar Escore_num y Cobertura_num)
by_proj = df_all.groupby("PRJ").agg(
    Cobertura_media=("Cobertura_num", "mean"),
    Escore_medio=("Escore_num", "mean"),
    Gerados_medios=("Gerados_num", "mean"),
    N_registros=("Temperature", "count")
).reset_index()

by_project_path = out_dir / "metrics_by_project.csv"
by_proj.to_csv(by_project_path, sep=";", index=False)
print(f"[ok] metrics_by_project.csv generado en {by_project_path}")

print("[done] Agregación finalizada correctamente")
