#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de gráficos para los resultados de la replicación exLong
Crea visualizaciones profesionales en formato SVG sin dependencias externas
"""

import os
import json

def cargar_datos(archivo):
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            return json.load(f)
    return None

def generar_svg_barras_horiz(datos, titulo, filename):
    """Genera un archivo SVG con un gráfico de barras horizontales"""
    if not datos:
        return
        
    labels = list(datos.keys())
    values = list(datos.values())
    
    width = 600
    height = len(labels) * 50 + 100
    bar_height = 30
    max_val = max(values) if values else 100
    scale = 400 / max_val if max_val > 0 else 1
    
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="100%" height="100%" fill="#f8f9fa"/>',
        f'<text x="{width/2}" y="40" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">{titulo}</text>'
    ]
    
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 80 + i * 50
        bar_w = val * scale
        svg.append(f'<text x="10" y="{y + 20}" font-family="Arial" font-size="14">{label}</text>')
        svg.append(f'<rect x="180" y="{y}" width="{bar_w}" height="{bar_height}" fill="#3498db"/>')
        svg.append(f'<text x="{180 + bar_w + 5}" y="{y + 20}" font-family="Arial" font-size="12" font-weight="bold">{val:.1f}</text>')
        
    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))

def generar_svg_barras_vert(datos, titulo, filename):
    """Genera un archivo SVG con un gráfico de barras verticales"""
    if not datos:
        return
        
    labels = list(datos.keys())
    values = list(datos.values())
    
    width = 600
    height = 400
    bar_width = 40
    gap = 20
    max_val = max(values) if values else 100
    scale = 250 / max_val if max_val > 0 else 1
    
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="100%" height="100%" fill="#f8f9fa"/>',
        f'<text x="{width/2}" y="40" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">{titulo}</text>'
    ]
    
    total_w = len(labels) * (bar_width + gap)
    start_x = (width - total_w) / 2
    
    for i, (label, val) in enumerate(zip(labels, values)):
        x = start_x + i * (bar_width + gap)
        h = val * scale
        y = height - 80 - h
        
        color = "#2ecc71" if "exLong" in label else "#e74c3c"
        
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="{color}"/>')
        svg.append(f'<text x="{x + bar_width/2}" y="{height - 60}" font-family="Arial" font-size="12" text-anchor="middle" transform="rotate(20 {x + bar_width/2} {height - 60})">{label}</text>')
        svg.append(f'<text x="{x + bar_width/2}" y="{y - 10}" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">{val:.1f}</text>')
        
    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))

def main():
    print("Generando gráficos de resultados (formato SVG)...")
    os.makedirs("resultados_replicacion", exist_ok=True)
    
    rq1_data = cargar_datos("resultados_replicacion/rq1_resumen.json")
    rq2_data = cargar_datos("resultados_replicacion/rq2_resumen.json")
    
    if rq1_data:
        generar_svg_barras_horiz(rq1_data, "RQ1: Developer-oriented Use Case", 'resultados_replicacion/grafico_rq1.svg')
        print("  - grafico_rq1.svg generado.")
    
    if rq2_data:
        generar_svg_barras_vert(rq2_data, "RQ2: Machine-oriented Use Case", 'resultados_replicacion/grafico_rq2.svg')
        print("  - grafico_rq2.svg generado.")

if __name__ == "__main__":
    main()
