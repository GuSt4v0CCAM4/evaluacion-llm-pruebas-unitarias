#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para ejecutar la replicación completa de exLong
Ejecuta los análisis equivalentes a RQ1 y RQ2 del paper
"""

import sys
import os

def main():
    """Ejecuta la replicación completa"""
    print("=" * 80)
    print("REPLICACIÓN DEL PROYECTO exLong")
    print("Generating Exceptional Behavior Tests with Large Language Models")
    print("=" * 80)
    
    # Crear directorio para resultados
    os.makedirs("resultados_replicacion", exist_ok=True)
    
    # RQ1: Developer-oriented Use Case
    print("\n\n")
    print("EJECUTANDO RQ1: Developer-oriented Use Case")
    print("=" * 80)
    try:
        import replicacion_rq1
        replicacion_rq1.analizar_rq1()
    except Exception as e:
        print(f"ERROR en RQ1: {e}")
        import traceback
        # traceback.print_exc()
    
    # RQ2: Machine-oriented Use Case
    print("\n\n")
    print("EJECUTANDO RQ2: Machine-oriented Use Case")
    print("=" * 80)
    try:
        import replicacion_rq2
        replicacion_rq2.analizar_rq2()
    except Exception as e:
        print(f"ERROR en RQ2: {e}")
        import traceback
        # traceback.print_exc()
    
    # Generación de Gráficos
    print("\n\n")
    print("GENERANDO GRÁFICOS")
    print("=" * 80)
    try:
        import generar_graficos
        generar_graficos.main()
    except Exception as e:
        print(f"ERROR al generar gráficos: {e}")

    print("\n\n")
    print("=" * 80)
    print("REPLICACIÓN FINALIZADA")
    print("=" * 80)
    print("\nRevisa los resultados en la carpeta 'resultados_replicacion/'")

if __name__ == "__main__":
    main()
