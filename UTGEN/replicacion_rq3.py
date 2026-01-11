#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicación RQ3: Análisis del Cuestionario Post-Test
Analiza qué elementos de UTGen afectan la comprensibilidad
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

def cohens_d(group1, group2):
    """Calcula Cohen's d para tamaño de efecto"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def parse_criteria_string(criteria_str):
    """Parsea string de criterios 'x, y, z, w' a lista de números"""
    if pd.isna(criteria_str) or criteria_str == '':
        return []
    try:
        return [float(x.strip()) for x in str(criteria_str).split(',')]
    except:
        return []

def analizar_rq3():
    """Analiza los resultados de RQ3"""
    print("=" * 80)
    print("REPLICACIÓN RQ3: Elementos que Afectan Comprensibilidad")
    print("=" * 80)
    
    # Cargar datos
    base_path = "UTGen/Results/RQ3"
    
    print("\nCargando datos desde:")
    criteria_file = os.path.join(base_path, "criteria.csv")
    priority_file = os.path.join(base_path, "priority.csv")
    q1_2_file = os.path.join(base_path, "Q1-2.csv")
    
    print(f"  - Criteria: {criteria_file}")
    print(f"  - Priority: {priority_file}")
    print(f"  - Q1-2: {q1_2_file}")
    
    try:
        criteria_df = pd.read_csv(criteria_file)
        priority_df = pd.read_csv(priority_file)
        q1_2_df = pd.read_csv(q1_2_file)
    except FileNotFoundError as e:
        print(f"ERROR: No se pudo encontrar el archivo: {e}")
        return
    
    print(f"\nDatos cargados:")
    print(f"  Criteria: {len(criteria_df)} registros")
    print(f"  Priority: {len(priority_df)} registros")
    print(f"  Q1-2: {len(q1_2_df)} registros")
    
    # Análisis de Q1-2 (Impacto de comprensibilidad)
    print("\n" + "=" * 80)
    print("Q1-Q2: Impacto de Comprensibilidad en Bug Fixing")
    print("=" * 80)
    
    # Q2: Importancia de comprensibilidad
    q2_col = 'Importance of understandability on bug fixing '
    if q2_col in q1_2_df.columns:
        q2_data = q1_2_df[q2_col].dropna()
        q2_data = pd.to_numeric(q2_data, errors='coerce').dropna()
        
        print(f"\nQ2 - Importancia de comprensibilidad (1-5):")
        print(f"  Media: {q2_data.mean():.2f}")
        print(f"  Mediana: {q2_data.median():.2f}")
        print(f"  Distribución:")
        for i in range(1, 6):
            count = (q2_data == i).sum()
            pct = count / len(q2_data) * 100
            print(f"    {i}: {count} ({pct:.1f}%)")
    
    # Análisis de Priority (Q3)
    print("\n" + "=" * 80)
    print("Q3: Priorización de Elementos")
    print("=" * 80)
    
    # Contar cuántas veces cada elemento está en primera posición
    if 'Prioritize' in priority_df.columns:
        priority_counts = {'Comment': 0, 'Test Name': 0, 'Variable Naming': 0, 'Test Data': 0}
        
        for _, row in priority_df.iterrows():
            priority_str = str(row['Prioritize'])
            if priority_str and priority_str != 'nan':
                # Obtener el primer elemento
                first = priority_str.split(',')[0].strip()
                if first in priority_counts:
                    priority_counts[first] += 1
        
        total = sum(priority_counts.values())
        print("\nElementos más importantes (Rank 1):")
        for element, count in sorted(priority_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {element}: {count} ({pct:.1f}%)")
    
    # Análisis de Criteria (Q6-Q7)
    print("\n" + "=" * 80)
    print("Q6-Q7: Criterios y Elementos")
    print("=" * 80)
    
    # Los criterios están en formato "x, y, z, w" donde:
    # x = completeness, y = conciseness, z = clarity, w = naturalness
    criteria_names = ['Completeness', 'Conciseness', 'Clarity', 'Naturalness']
    element_names = ['Comments', 'Test Data', 'Test Name', 'Variable Names']
    
    # Extraer valores por elemento
    results = {}
    
    for element in element_names:
        element_col = element
        if element_col in criteria_df.columns:
            utg_values = []
            evo_values = []
            
            for _, row in criteria_df.iterrows():
                test_gen = row['TestGen']
                criteria_str = row[element_col]
                values = parse_criteria_string(criteria_str)
                
                if len(values) == 4:  # completeness, conciseness, clarity, naturalness
                    if test_gen == 'UTG':
                        utg_values.append(values)
                    elif test_gen == 'Evo':
                        evo_values.append(values)
            
            if utg_values and evo_values:
                utg_array = np.array(utg_values)
                evo_array = np.array(evo_values)
                
                results[element] = {
                    'UTG': utg_array,
                    'Evo': evo_array
                }
    
    # Análisis por criterio
    print("\n" + "-" * 80)
    print("Análisis por Criterio (Completeness, Conciseness, Clarity, Naturalness)")
    print("-" * 80)
    
    for idx, criterion in enumerate(criteria_names):
        print(f"\n{criterion}:")
        utg_all = []
        evo_all = []
        
        for element, data in results.items():
            utg_criterion = data['UTG'][:, idx].flatten()
            evo_criterion = data['Evo'][:, idx].flatten()
            utg_all.extend(utg_criterion)
            evo_all.extend(evo_criterion)
        
        if utg_all and evo_all:
            utg_array = np.array(utg_all)
            evo_array = np.array(evo_all)
            
            print(f"  UTGen: Media={utg_array.mean():.2f}, Mediana={np.median(utg_array):.2f}")
            print(f"  EvoSuite: Media={evo_array.mean():.2f}, Mediana={np.median(evo_array):.2f}")
            
            # Wilcoxon test
            stat, p_value = wilcoxon(utg_array, evo_array)
            print(f"  Wilcoxon test: p={p_value:.4f}")
            
            d = cohens_d(utg_array, evo_array)
            print(f"  Cohen's d: {d:.2f}")
    
    # Análisis por elemento
    print("\n" + "-" * 80)
    print("Análisis por Elemento")
    print("-" * 80)
    
    for element, data in results.items():
        print(f"\n{element}:")
        utg_all = data['UTG'].flatten()
        evo_all = data['Evo'].flatten()
        
        print(f"  UTGen: Media={utg_all.mean():.2f}, Mediana={np.median(utg_all):.2f}")
        print(f"  EvoSuite: Media={evo_all.mean():.2f}, Mediana={np.median(evo_all):.2f}")
        
        # Wilcoxon test
        stat, p_value = wilcoxon(utg_all, evo_all)
        print(f"  Wilcoxon test: p={p_value:.4f}")
        
        d = cohens_d(utg_all, evo_all)
        print(f"  Cohen's d: {d:.2f}")
    
    # Comparación con el paper
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON EL PAPER")
    print("=" * 80)
    print("\nSegún el paper:")
    print("  Q2: Mediana 4/5 de acuerdo sobre importancia")
    print("  Q3: 34.3% priorizan Comment, 40.6% priorizan Test Name")
    print("  Q6-Q7: Tests UTGen mejor evaluados en todos los criterios")
    print("    - Completeness: d grande (>0.8)")
    print("    - Clarity: d grande (>0.8)")
    print("    - Naturalness: d grande (>0.8)")
    print("    - Conciseness: d pequeño")
    print("  Variable naming: d muy grande (>1.2)")
    
    # Guardar resultados
    output_dir = "resultados_replicacion"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nResultados guardados en: {output_dir}/")
    
    return criteria_df, priority_df, q1_2_df

if __name__ == "__main__":
    analizar_rq3()

