"""Ejecuta el notebook en su propio kernel y conserva sus salidas."""
from pathlib import Path
import argparse
import json
import os
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prueba", action="store_true", help="300 train/30 eval, dos épocas; no genera entregas de competencia")
    parser.add_argument("--notebook", type=Path, default=Path("proyecto2.ipynb"))
    args = parser.parse_args()
    if args.prueba and args.notebook.name != "proyecto2.ipynb":
        parser.error("--prueba se aplica únicamente al notebook inicial proyecto2.ipynb")
    raiz = Path(__file__).resolve().parents[1]
    os.chdir(raiz)
    if args.prueba:
        os.environ["P2_MODO_PRUEBA"] = "1"
    else:
        os.environ.pop("P2_MODO_PRUEBA", None)
    os.environ["MPLCONFIGDIR"] = str(raiz / "tmp/matplotlib")
    os.environ["IPYTHONDIR"] = str(raiz / "tmp/ipython")
    notebook = nbformat.read(raiz / args.notebook, as_version=4)
    destino = raiz / "output/parte2/pruebas/proyecto2_prueba.ipynb" if args.prueba else raiz / args.notebook
    destino.parent.mkdir(parents=True, exist_ok=True)
    kernel = KernelManager(kernel_name="python3")
    # Ejecutar con el mismo entorno desde el que se llamó este script.
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    print("Iniciando kernel Python:", sys.executable, flush=True)
    cliente = NotebookClient(notebook, km=kernel, timeout=None, resources={"metadata": {"path": str(raiz)}},
                             on_cell_start=lambda cell, cell_index: print(f"Celda {cell_index + 1}/{len(notebook.cells)}", flush=True))
    try:
        cliente.execute()
    finally:
        nbformat.write(notebook, destino)
        print("Notebook guardado:", destino, flush=True)


if __name__ == "__main__":
    main()
