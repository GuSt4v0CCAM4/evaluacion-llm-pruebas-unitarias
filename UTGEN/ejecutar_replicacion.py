#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para ejecutar la replicación completa del paper UTGen
Ejecuta los análisis de RQ1, RQ2 y RQ3
"""

import sys
import os

def main():
    """Ejecuta la replicación completa"""
    print("=" * 80)
    print("REPLICACIÓN DEL PAPER UTGen")
    print("Leveraging Large Language Models for Enhancing")
    print("the Understandability of Generated Unit Tests")
    print("=" * 80)
    
    # Crear directorio para resultados
    os.makedirs("resultados_replicacion", exist_ok=True)
    
    # RQ1: Efectividad
    print("\n\n")
    print("EJECUTANDO RQ1: Efectividad de UTGen")
    print("=" * 80)
    try:
        import replicacion_rq1
        replicacion_rq1.analizar_rq1()
    except Exception as e:
        print(f"ERROR en RQ1: {e}")
        import traceback
        traceback.print_exc()
    
    # RQ2: Bug Fixing
    print("\n\n")
    print("EJECUTANDO RQ2: Impacto en Bug Fixing")
    print("=" * 80)
    try:
        import replicacion_rq2
        replicacion_rq2.analizar_rq2()
    except Exception as e:
        print(f"ERROR en RQ2: {e}")
        import traceback
        traceback.print_exc()
    
    # RQ3: Cuestionario
    print("\n\n")
    print("EJECUTANDO RQ3: Elementos de Comprensibilidad")
    print("=" * 80)
    try:
        import replicacion_rq3
        replicacion_rq3.analizar_rq3()
    except Exception as e:
        print(f"ERROR en RQ3: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n")
    print("=" * 80)
    print("REPLICACIÓN COMPLETA")
    print("=" * 80)
    print("\nTodos los análisis han sido ejecutados.")
    print("Revisa los resultados en la carpeta 'resultados_replicacion/'")
    print("\nNOTA: Para análisis estadísticos completos (Mixed Models) se recomienda")
    print("usar R con los paquetes 'ordinal' y 'lme4' como se describe en el paper.")

if __name__ == "__main__":
    main()

