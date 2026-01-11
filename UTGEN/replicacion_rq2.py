#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicación RQ2: Análisis del Experimento Controlado (Bug Fixing)
Analiza el impacto de tests mejorados por LLM en bug fixing
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def parse_time(time_str):
    """Convierte tiempo en formato HH:MM:SS a minutos"""
    if pd.isna(time_str) or time_str == '':
        return np.nan
    
    parts = str(time_str).split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 60 + minutes + seconds / 60
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes + seconds / 60
    else:
        return np.nan

def cohens_d(group1, group2):
    """Calcula Cohen's d para tamaño de efecto"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def analizar_rq2():
    """Analiza los resultados de RQ2"""
    print("=" * 80)
    print("REPLICACIÓN RQ2: Impacto en Bug Fixing")
    print("=" * 80)
    
    # Cargar datos
    results_file = "UTGen/Results/RQ2/results.csv"
    print(f"\nCargando datos desde: {results_file}")
    
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"ERROR: No se pudo encontrar el archivo: {results_file}")
        return
    
    print(f"Datos cargados: {len(df)} registros")
    
    # Preparar datos
    df['time_minutes'] = df['time'].apply(parse_time)
    
    # Renombrar columnas para facilitar acceso
    df.columns = df.columns.str.strip()
    
    print("\n" + "-" * 80)
    print("ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 80)
    
    # Bugs arreglados por técnica
    print("\nBUGS ARREGLADOS:")
    for technique in ['UTG', 'Evo']:
        tech_data = df[df['TestGen'] == technique]
        bugs = tech_data['# bugs fixed']
        print(f"\n  {technique}:")
        print(f"    Media: {bugs.mean():.2f}")
        print(f"    Mediana: {bugs.median():.2f}")
        print(f"    Desv. Est.: {bugs.std():.2f}")
        print(f"    Mín: {bugs.min()}, Máx: {bugs.max()}")
    
    # Tiempo por técnica
    print("\nTIEMPO (minutos):")
    for technique in ['UTG', 'Evo']:
        tech_data = df[df['TestGen'] == technique]
        time_data = tech_data['time_minutes'].dropna()
        print(f"\n  {technique}:")
        print(f"    Media: {time_data.mean():.2f} min")
        print(f"    Mediana: {time_data.median():.2f} min")
        print(f"    Desv. Est.: {time_data.std():.2f} min")
    
    # Análisis por objeto
    print("\n" + "-" * 80)
    print("ANÁLISIS POR OBJETO")
    print("-" * 80)
    
    for obj in ['Budget', 'JSWeaponData']:
        obj_data = df[df['Task'] == obj]
        print(f"\n{obj}:")
        for technique in ['UTG', 'Evo']:
            tech_obj = obj_data[obj_data['TestGen'] == technique]
            bugs = tech_obj['# bugs fixed']
            time_data = tech_obj['time_minutes'].dropna()
            print(f"  {technique}:")
            print(f"    Bugs arreglados - Media: {bugs.mean():.2f}, Mediana: {bugs.median():.2f}")
            if len(time_data) > 0:
                print(f"    Tiempo - Media: {time_data.mean():.2f} min, Mediana: {time_data.median():.2f} min")
    
    # Tests estadísticos simples (Wilcoxon para muestras relacionadas no es adecuado aquí
    # ya que cada participante tiene dos tareas diferentes)
    print("\n" + "-" * 80)
    print("COMPARACIÓN ESTADÍSTICA SIMPLE")
    print("-" * 80)
    print("\nNOTA: El paper usa Mixed Models (Cumulative Link Mixed Models para bugs,")
    print("Generalized Linear Mixed Models para tiempo) debido al diseño crossover.")
    print("Aquí mostramos comparaciones simples para referencia.\n")
    
    # Comparación de bugs arreglados
    utg_bugs = df[df['TestGen'] == 'UTG']['# bugs fixed']
    evo_bugs = df[df['TestGen'] == 'Evo']['# bugs fixed']
    
    # Mann-Whitney U test (para muestras independientes, aproximación)
    stat_bugs, p_bugs = stats.mannwhitneyu(utg_bugs, evo_bugs, alternative='two-sided')
    print(f"Bugs Arreglados (Mann-Whitney U):")
    print(f"  Estadístico: {stat_bugs:.2f}")
    print(f"  p-value: {p_bugs:.4f}")
    d_bugs = cohens_d(utg_bugs, evo_bugs)
    print(f"  Cohen's d: {d_bugs:.2f}")
    
    # Comparación de tiempo
    utg_time = df[df['TestGen'] == 'UTG']['time_minutes'].dropna()
    evo_time = df[df['TestGen'] == 'Evo']['time_minutes'].dropna()
    
    if len(utg_time) > 0 and len(evo_time) > 0:
        stat_time, p_time = stats.mannwhitneyu(utg_time, evo_time, alternative='two-sided')
        print(f"\nTiempo (Mann-Whitney U):")
        print(f"  Estadístico: {stat_time:.2f}")
        print(f"  p-value: {p_time:.4f}")
        d_time = cohens_d(utg_time, evo_time)
        print(f"  Cohen's d: {d_time:.2f}")
    
    # Comparación con el paper
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON EL PAPER")
    print("=" * 80)
    print("\nSegún el paper (Mixed Models):")
    print("  Bugs arreglados:")
    print("    Technique: p = 0.024 (significativo)")
    print("    Object: p = 0.025 (significativo)")
    print("    Cohen's d: 0.59 (efecto mediano)")
    print("  Tiempo:")
    print("    Technique: p = 0.063 (no significativo)")
    print("    Object: p = 0.031 (significativo)")
    
    # Guardar resultados
    output_dir = "resultados_replicacion"
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(f"{output_dir}/rq2_datos_procesados.csv", index=False)
    
    # Resumen estadístico
    summary = {
        'Metrica': ['Bugs Arreglados - UTG', 'Bugs Arreglados - Evo', 
                   'Tiempo (min) - UTG', 'Tiempo (min) - Evo'],
        'Media': [utg_bugs.mean(), evo_bugs.mean(), 
                 utg_time.mean() if len(utg_time) > 0 else np.nan,
                 evo_time.mean() if len(evo_time) > 0 else np.nan],
        'Mediana': [utg_bugs.median(), evo_bugs.median(),
                   utg_time.median() if len(utg_time) > 0 else np.nan,
                   evo_time.median() if len(evo_time) > 0 else np.nan]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/rq2_resumen.csv", index=False)
    
    print(f"\nResultados guardados en: {output_dir}/")
    
    return df

if __name__ == "__main__":
    analizar_rq2()

