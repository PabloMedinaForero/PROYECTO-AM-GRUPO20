"""Convierte las celdas marcadas del script en un notebook autónomo para Kaggle."""
from pathlib import Path
import re

import nbformat


def main():
    raiz = Path(__file__).resolve().parents[1]
    fuente = (raiz / "scripts/finetuning_parte2.py").read_text(encoding="utf-8")
    partes = re.split(r"^# %%([^\n]*)\n", fuente, flags=re.M)
    celdas = []
    for i in range(1, len(partes), 2):
        etiqueta, texto = partes[i].strip(), partes[i + 1].rstrip() + "\n"
        if not texto.strip():
            continue
        if "[markdown]" in etiqueta:
            texto = "\n".join(linea[2:] if linea.startswith("# ") else "" if linea == "#" else linea
                              for linea in texto.splitlines())
            celdas.append(nbformat.v4.new_markdown_cell(texto))
        else:
            compile(texto, f"celda-{len(celdas)}", "exec")
            celdas.append(nbformat.v4.new_code_cell(texto))
    analisis = raiz / "output/parte2/mejora/analisis_validacion.md"
    if analisis.exists():
        celdas.append(nbformat.v4.new_markdown_cell(analisis.read_text(encoding="utf-8")))
    notebook = nbformat.v4.new_notebook(cells=celdas)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.kaggle = {"accelerator": "gpu", "isInternetEnabled": True,
                                "language": "python", "sourceType": "notebook", "isGpuEnabled": True}
    nbformat.validate(notebook)
    nbformat.write(notebook, raiz / "proyecto2_mejorado.ipynb")
    print("Notebook autónomo generado:", len(celdas), "celdas")


if __name__ == "__main__":
    main()
