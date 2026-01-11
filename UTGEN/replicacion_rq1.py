#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicación RQ1: Análisis de Efectividad de UTGen
Compara UTGen vs EvoSuite en términos de cobertura y tasas de compilación
"""

import pandas as pd
import numpy as np
import os

def calcular_cobertura(row):
    """Calcula el porcentaje de cobertura"""
    if row['INSTRUCTION_COVERED'] + row['INSTRUCTION_MISSED'] == 0:
        return 0
    instruction_coverage = row['INSTRUCTION_COVERED'] / (row['INSTRUCTION_COVERED'] + row['INSTRUCTION_MISSED']) * 100
    
    if row['BRANCH_COVERED'] + row['BRANCH_MISSED'] == 0:
        branch_coverage = 0
    else:
        branch_coverage = row['BRANCH_COVERED'] / (row['BRANCH_COVERED'] + row['BRANCH_MISSED']) * 100
    
    return instruction_coverage, branch_coverage

def analizar_rq1():
    """Analiza los resultados de RQ1"""
    print("=" * 80)
    print("REPLICACIÓN RQ1: Efectividad de UTGen")
    print("=" * 80)
    
    # Cargar datos
    base_path = "UTGen/Results/RQ1"
    evosuite_file = os.path.join(base_path, "aggregated_results_ES.csv")
    utgen_file = os.path.join(base_path, "aggregated_results_UTG.csv")
    
    print(f"\nCargando datos desde:")
    print(f"  - EvoSuite: {evosuite_file}")
    print(f"  - UTGen: {utgen_file}")
    
    try:
        evosuite_df = pd.read_csv(evosuite_file)
        utgen_df = pd.read_csv(utgen_file)
    except FileNotFoundError as e:
        print(f"ERROR: No se pudo encontrar el archivo: {e}")
        return
    
    print(f"\nEvoSuite: {len(evosuite_df)} registros")
    print(f"UTGen: {len(utgen_df)} registros")
    
    # Calcular cobertura
    print("\n" + "-" * 80)
    print("CÁLCULO DE COBERTURA")
    print("-" * 80)
    
    # EvoSuite
    evosuite_inst = []
    evosuite_branch = []
    for _, row in evosuite_df.iterrows():
        inst_cov, branch_cov = calcular_cobertura(row)
        evosuite_inst.append(inst_cov)
        evosuite_branch.append(branch_cov)
    
    evosuite_df['INSTRUCTION_COVERAGE_PCT'] = evosuite_inst
    evosuite_df['BRANCH_COVERAGE_PCT'] = evosuite_branch
    
    # UTGen
    utgen_inst = []
    utgen_branch = []
    for _, row in utgen_df.iterrows():
        inst_cov, branch_cov = calcular_cobertura(row)
        utgen_inst.append(inst_cov)
        utgen_branch.append(branch_cov)
    
    utgen_df['INSTRUCTION_COVERAGE_PCT'] = utgen_inst
    utgen_df['BRANCH_COVERAGE_PCT'] = utgen_branch
    
    # Estadísticas descriptivas
    print("\nCOBERTURA DE INSTRUCCIONES:")
    print(f"  EvoSuite:")
    print(f"    Media: {np.mean(evosuite_inst):.2f}%")
    print(f"    Mediana: {np.median(evosuite_inst):.2f}%")
    print(f"    Desv. Est.: {np.std(evosuite_inst):.2f}%")
    
    print(f"  UTGen:")
    print(f"    Media: {np.mean(utgen_inst):.2f}%")
    print(f"    Mediana: {np.median(utgen_inst):.2f}%")
    print(f"    Desv. Est.: {np.std(utgen_inst):.2f}%")
    
    print("\nCOBERTURA DE RAMAS:")
    print(f"  EvoSuite:")
    print(f"    Media: {np.mean(evosuite_branch):.2f}%")
    print(f"    Mediana: {np.median(evosuite_branch):.2f}%")
    print(f"    Desv. Est.: {np.std(evosuite_branch):.2f}%")
    
    print(f"  UTGen:")
    print(f"    Media: {np.mean(utgen_branch):.2f}%")
    print(f"    Mediana: {np.median(utgen_branch):.2f}%")
    print(f"    Desv. Est.: {np.std(utgen_branch):.2f}%")
    
    # Comparación con el paper
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON EL PAPER")
    print("=" * 80)
    print("\nSegún el paper:")
    print("  EvoSuite:")
    print("    Cobertura Instrucciones: 25.03%")
    print("    Cobertura Ramas: 18.68%")
    print("  UTGen:")
    print("    Cobertura Instrucciones: 24.43%")
    print("    Cobertura Ramas: 17.87%")
    
    print("\nNuestros resultados:")
    print(f"  EvoSuite:")
    print(f"    Cobertura Instrucciones: {np.mean(evosuite_inst):.2f}%")
    print(f"    Cobertura Ramas: {np.mean(evosuite_branch):.2f}%")
    print(f"  UTGen:")
    print(f"    Cobertura Instrucciones: {np.mean(utgen_inst):.2f}%")
    print(f"    Cobertura Ramas: {np.mean(utgen_branch):.2f}%")
    
    # Nota sobre pass rate
    print("\n" + "-" * 80)
    print("NOTA SOBRE PASS RATE")
    print("-" * 80)
    print("El paper menciona:")
    print("  - UTGen: 8430 tests generados, 73.27% pass rate")
    print("  - EvoSuite: 8315 tests generados, 79.01% pass rate")
    print("\nEstos datos no están directamente en los CSV de cobertura.")
    print("Se necesitarían los logs completos de ejecución para calcularlos.")
    
    # Guardar resultados procesados
    output_dir = "resultados_replicacion"
    os.makedirs(output_dir, exist_ok=True)
    
    evosuite_df.to_csv(f"{output_dir}/evosuite_con_cobertura.csv", index=False)
    utgen_df.to_csv(f"{output_dir}/utgen_con_cobertura.csv", index=False)
    
    print(f"\nResultados guardados en: {output_dir}/")
    
    return evosuite_df, utgen_df

if __name__ == "__main__":
    analizar_rq1()

