#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de gráficos para los resultados de la replicación UTGen
Crea visualizaciones profesionales de los resultados de RQ2 y RQ3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Intentar importar seaborn (opcional)
try:
    import seaborn as sns
    sns.set_palette("husl")
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False

# Configurar estilo
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

# Configurar matplotlib para usar backend sin display (para servidor)
plt.switch_backend('Agg')

def parse_time_correct(time_str):
    """Convierte tiempo en formato HH:MM:SS o MM:SS a minutos"""
    if pd.isna(time_str) or time_str == '':
        return np.nan
    
    parts = str(time_str).split(':')
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 60 + minutes + seconds / 60
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes + seconds / 60
    else:
        return np.nan

def cohens_d(group1, group2):
    """Calcula Cohen's d"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def generar_graficos_rq2():
    """Genera gráficos para RQ2: Bug Fixing"""
    print("Generando graficos RQ2...")
    
    # Cargar datos
    df = pd.read_csv("UTGen/Results/RQ2/results.csv")
    df['time_minutes'] = df['time'].apply(parse_time_correct)
    df['Tecnica'] = df['TestGen'].map({'UTG': 'UTGen', 'Evo': 'EvoSuite'})
    
    # Crear directorio para gráficos
    os.makedirs("resultados_replicacion/graficos", exist_ok=True)
    
    # ===== GRÁFICO 1: Bugs Arreglados (Box Plot) =====
    fig, ax = plt.subplots(figsize=(10, 6))
    data_bugs = [df[df['Tecnica'] == 'UTGen']['# bugs fixed'].values,
                 df[df['Tecnica'] == 'EvoSuite']['# bugs fixed'].values]
    
    bp = ax.boxplot(data_bugs, labels=['UTGen', 'EvoSuite'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][1].set_facecolor('#F44336')
    
    # Agregar puntos individuales
    for i, tec in enumerate(['UTGen', 'EvoSuite']):
        y = df[df['Tecnica'] == tec]['# bugs fixed'].values
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.plot(x, y, 'ro', alpha=0.5, markersize=4)
    
    ax.set_ylabel('Bugs Arreglados (de 4)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tecnica', fontsize=12, fontweight='bold')
    ax.set_title('RQ2: Bugs Arreglados - UTGen vs EvoSuite', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, 4.5)
    ax.grid(True, alpha=0.3)
    
    # Agregar estadísticas
    utg_mean = df[df['Tecnica'] == 'UTGen']['# bugs fixed'].mean()
    evo_mean = df[df['Tecnica'] == 'EvoSuite']['# bugs fixed'].mean()
    utg_std = df[df['Tecnica'] == 'UTGen']['# bugs fixed'].std()
    evo_std = df[df['Tecnica'] == 'EvoSuite']['# bugs fixed'].std()
    
    stat, p_value = stats.mannwhitneyu(df[df['Tecnica'] == 'UTGen']['# bugs fixed'],
                                       df[df['Tecnica'] == 'EvoSuite']['# bugs fixed'])
    d = cohens_d(df[df['Tecnica'] == 'UTGen']['# bugs fixed'],
                 df[df['Tecnica'] == 'EvoSuite']['# bugs fixed'])
    
    textstr = f'UTGen: μ={utg_mean:.2f}, σ={utg_std:.2f}\nEvoSuite: μ={evo_mean:.2f}, σ={evo_std:.2f}\np={p_value:.4f}, d={d:.2f}'
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('resultados_replicacion/graficos/RQ2_bugs_arreglados.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== GRÁFICO 2: Tiempo (Box Plot) =====
    fig, ax = plt.subplots(figsize=(10, 6))
    data_time = [df[df['Tecnica'] == 'UTGen']['time_minutes'].dropna().values,
                 df[df['Tecnica'] == 'EvoSuite']['time_minutes'].dropna().values]
    
    bp = ax.boxplot(data_time, labels=['UTGen', 'EvoSuite'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][1].set_facecolor('#F44336')
    
    # Agregar puntos individuales
    for i, tec in enumerate(['UTGen', 'EvoSuite']):
        y = df[df['Tecnica'] == tec]['time_minutes'].dropna().values
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.plot(x, y, 'ro', alpha=0.5, markersize=4)
    
    ax.set_ylabel('Tiempo (minutos)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tecnica', fontsize=12, fontweight='bold')
    ax.set_title('RQ2: Tiempo para Arreglar Bugs - UTGen vs EvoSuite', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Agregar estadísticas
    utg_time_mean = df[df['Tecnica'] == 'UTGen']['time_minutes'].mean()
    evo_time_mean = df[df['Tecnica'] == 'EvoSuite']['time_minutes'].mean()
    
    stat, p_value = stats.mannwhitneyu(df[df['Tecnica'] == 'UTGen']['time_minutes'].dropna(),
                                       df[df['Tecnica'] == 'EvoSuite']['time_minutes'].dropna())
    d = cohens_d(df[df['Tecnica'] == 'UTGen']['time_minutes'].dropna(),
                 df[df['Tecnica'] == 'EvoSuite']['time_minutes'].dropna())
    
    textstr = f'UTGen: μ={utg_time_mean:.1f} min\nEvoSuite: μ={evo_time_mean:.1f} min\np={p_value:.4f}, d={d:.2f}'
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('resultados_replicacion/graficos/RQ2_tiempo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== GRÁFICO 3: Comparación de Medias (Bar Plot) =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bugs arreglados
    bugs_data = {
        'UTGen': utg_mean,
        'EvoSuite': evo_mean
    }
    colors = ['#4CAF50', '#F44336']
    bars1 = ax1.bar(bugs_data.keys(), bugs_data.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.errorbar(bugs_data.keys(), bugs_data.values(), 
                yerr=[utg_std, evo_std], fmt='none', color='black', capsize=10, capthick=2)
    ax1.set_ylabel('Bugs Arreglados (Media)', fontsize=12, fontweight='bold')
    ax1.set_title('Bugs Arreglados - Comparacion de Medias', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 4.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, val in zip(bars1, bugs_data.values()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Tiempo
    time_data = {
        'UTGen': utg_time_mean,
        'EvoSuite': evo_time_mean
    }
    bars2 = ax2.bar(time_data.keys(), time_data.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Tiempo (minutos)', fontsize=12, fontweight='bold')
    ax2.set_title('Tiempo para Arreglar Bugs - Comparacion de Medias', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, val in zip(bars2, time_data.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('resultados_replicacion/graficos/RQ2_comparacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  - RQ2_bugs_arreglados.png")
    print("  - RQ2_tiempo.png")
    print("  - RQ2_comparacion.png")

def generar_graficos_rq3():
    """Genera gráficos para RQ3: Comprensibilidad"""
    print("Generando graficos RQ3...")
    
    criteria_df = pd.read_csv("UTGen/Results/RQ3/criteria.csv")
    
    def parse_criteria(criteria_str):
        if pd.isna(criteria_str) or criteria_str == '':
            return []
        try:
            return [float(x.strip()) for x in str(criteria_str).split(',')]
        except:
            return []
    
    criteria_names = ['Completeness', 'Conciseness', 'Clarity', 'Naturalness']
    element_names = ['Comments', 'Test Data', 'Test Name', 'Variable Names']
    
    # Extraer datos
    results_criteria = {}
    results_elements = {}
    
    for idx, criterion in enumerate(criteria_names):
        utg_all = []
        evo_all = []
        for element in element_names:
            if element in criteria_df.columns:
                for _, row in criteria_df.iterrows():
                    test_gen = row['TestGen']
                    criteria_str = row[element]
                    values = parse_criteria(criteria_str)
                    if len(values) == 4:
                        if test_gen == 'UTG':
                            utg_all.append(values[idx])
                        elif test_gen == 'Evo':
                            evo_all.append(values[idx])
        results_criteria[criterion] = {'UTGen': np.array(utg_all), 'EvoSuite': np.array(evo_all)}
    
    for element in element_names:
        if element in criteria_df.columns:
            utg_all = []
            evo_all = []
            for _, row in criteria_df.iterrows():
                test_gen = row['TestGen']
                criteria_str = row[element]
                values = parse_criteria(criteria_str)
                if len(values) == 4:
                    if test_gen == 'UTG':
                        utg_all.extend(values)
                    elif test_gen == 'Evo':
                        evo_all.extend(values)
            results_elements[element] = {'UTGen': np.array(utg_all), 'EvoSuite': np.array(evo_all)}
    
    # ===== GRÁFICO 1: Criterios (Bar Plot) =====
    fig, ax = plt.subplots(figsize=(12, 6))
    
    utg_means = [results_criteria[c]['UTGen'].mean() for c in criteria_names]
    evo_means = [results_criteria[c]['EvoSuite'].mean() for c in criteria_names]
    utg_stds = [results_criteria[c]['UTGen'].std() for c in criteria_names]
    evo_stds = [results_criteria[c]['EvoSuite'].std() for c in criteria_names]
    
    x = np.arange(len(criteria_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, utg_means, width, label='UTGen', color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, evo_means, width, label='EvoSuite', color='#F44336', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.errorbar(x - width/2, utg_means, yerr=utg_stds, fmt='none', color='black', capsize=5)
    ax.errorbar(x + width/2, evo_means, yerr=evo_stds, fmt='none', color='black', capsize=5)
    
    ax.set_ylabel('Calificacion Media (1-5)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Criterio', fontsize=12, fontweight='bold')
    ax.set_title('RQ3: Evaluacion por Criterio - UTGen vs EvoSuite', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_names)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('resultados_replicacion/graficos/RQ3_criterios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== GRÁFICO 2: Elementos (Bar Plot) =====
    fig, ax = plt.subplots(figsize=(12, 6))
    
    utg_elem_means = [results_elements[e]['UTGen'].mean() for e in element_names]
    evo_elem_means = [results_elements[e]['EvoSuite'].mean() for e in element_names]
    utg_elem_stds = [results_elements[e]['UTGen'].std() for e in element_names]
    evo_elem_stds = [results_elements[e]['EvoSuite'].std() for e in element_names]
    
    x = np.arange(len(element_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, utg_elem_means, width, label='UTGen', color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, evo_elem_means, width, label='EvoSuite', color='#F44336', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.errorbar(x - width/2, utg_elem_means, yerr=utg_elem_stds, fmt='none', color='black', capsize=5)
    ax.errorbar(x + width/2, evo_elem_means, yerr=evo_elem_stds, fmt='none', color='black', capsize=5)
    
    ax.set_ylabel('Calificacion Media (1-5)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Elemento', fontsize=12, fontweight='bold')
    ax.set_title('RQ3: Evaluacion por Elemento - UTGen vs EvoSuite', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(element_names, rotation=15, ha='right')
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('resultados_replicacion/graficos/RQ3_elementos.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== GRÁFICO 3: Heatmap de Criterios por Elemento =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # UTGen
    heatmap_data_utg = []
    for element in element_names:
        row = []
        for criterion in criteria_names:
            utg_values = []
            if element in criteria_df.columns:
                idx = criteria_names.index(criterion)
                for _, row_data in criteria_df.iterrows():
                    if row_data['TestGen'] == 'UTG':
                        values = parse_criteria(row_data[element])
                        if len(values) == 4:
                            utg_values.append(values[idx])
            row.append(np.mean(utg_values) if utg_values else 0)
        heatmap_data_utg.append(row)
    
    # EvoSuite
    heatmap_data_evo = []
    for element in element_names:
        row = []
        for criterion in criteria_names:
            evo_values = []
            if element in criteria_df.columns:
                idx = criteria_names.index(criterion)
                for _, row_data in criteria_df.iterrows():
                    if row_data['TestGen'] == 'Evo':
                        values = parse_criteria(row_data[element])
                        if len(values) == 4:
                            evo_values.append(values[idx])
            row.append(np.mean(evo_values) if evo_values else 0)
        heatmap_data_evo.append(row)
    
    im1 = ax1.imshow(heatmap_data_utg, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    ax1.set_xticks(range(len(criteria_names)))
    ax1.set_yticks(range(len(element_names)))
    ax1.set_xticklabels(criteria_names, rotation=45, ha='right')
    ax1.set_yticklabels(element_names)
    ax1.set_title('UTGen', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Calificacion (1-5)')
    
    # Agregar valores
    for i in range(len(element_names)):
        for j in range(len(criteria_names)):
            text = ax1.text(j, i, f'{heatmap_data_utg[i][j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    im2 = ax2.imshow(heatmap_data_evo, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    ax2.set_xticks(range(len(criteria_names)))
    ax2.set_yticks(range(len(element_names)))
    ax2.set_xticklabels(criteria_names, rotation=45, ha='right')
    ax2.set_yticklabels(element_names)
    ax2.set_title('EvoSuite', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Calificacion (1-5)')
    
    # Agregar valores
    for i in range(len(element_names)):
        for j in range(len(criteria_names)):
            text = ax2.text(j, i, f'{heatmap_data_evo[i][j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    fig.suptitle('RQ3: Heatmap de Criterios por Elemento', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('resultados_replicacion/graficos/RQ3_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  - RQ3_criterios.png")
    print("  - RQ3_elementos.png")
    print("  - RQ3_heatmap.png")

def main():
    """Genera todos los gráficos"""
    print("=" * 80)
    print("GENERANDO GRAFICOS DE RESULTADOS - REPLICACION UTGen")
    print("=" * 80)
    print()
    
    try:
        generar_graficos_rq2()
        print()
        generar_graficos_rq3()
        
        print()
        print("=" * 80)
        print("GRAFICOS GENERADOS EXITOSAMENTE")
        print("=" * 80)
        print("\nTodos los graficos se encuentran en: resultados_replicacion/graficos/")
        print("\nArchivos generados:")
        print("  RQ2:")
        print("    - RQ2_bugs_arreglados.png (Box plot)")
        print("    - RQ2_tiempo.png (Box plot)")
        print("    - RQ2_comparacion.png (Barras comparativas)")
        print("  RQ3:")
        print("    - RQ3_criterios.png (Barras por criterio)")
        print("    - RQ3_elementos.png (Barras por elemento)")
        print("    - RQ3_heatmap.png (Heatmap comparativo)")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

