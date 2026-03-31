#!/usr/bin/env python3
"""
Script para ejecutar el notebook completo con manejo de timeout
"""
import subprocess
import sys

notebook_path = "GUARDIANAI_Hito2_3.ipynb"
output_path = "GUARDIANAI_Hito2_3_exec.ipynb"

print("="*70)
print("EJECUTANDO NOTEBOOK COMPLETO")
print("="*70)
print(f"Input:  {notebook_path}")
print(f"Output: {output_path}")
print(f"Timeout: 3600 segundos (1 hora)")
print("="*70)

try:
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            notebook_path,
            f"--output={output_path}",
            "--ExecutePreprocessor.timeout=3600"
        ],
        cwd=".",
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ EJECUCIÓN COMPLETADA EXITOSAMENTE")
    else:
        print(f"\n❌ Error durante la ejecución (código: {result.returncode})")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    sys.exit(1)
