"""Reproduce el nuevo CSV sin entrenar ni consultar train.csv."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import finetuning_parte2 as ft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=Path("output/parte2/mejora/entrega"))
    parser.add_argument("--eval", type=Path, default=Path("data/parte2/eval.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/parte2/sample_submission.csv"))
    parser.add_argument("--salida", type=Path, default=Path("output/parte2/mejora/submission_reproducida.csv"))
    args = parser.parse_args()
    tokenizador = ft.AutoTokenizer.from_pretrained(args.modelo / "encoder", local_files_only=True)
    modelo = ft.cargar_modelo(args.modelo)
    datos, sample = pd.read_csv(args.eval), pd.read_csv(args.sample)
    if not datos["id"].is_unique or not sample["id"].is_unique or set(datos["id"]) != set(sample["id"]):
        raise ValueError("Los IDs de eval y sample deben coincidir sin duplicados.")
    probs, _, _ = ft.evaluar_modelo(modelo, ft.crear_lotes(datos, tokenizador, con_etiquetas=False), con_etiquetas=False)
    por_id = pd.Series(np.array(ft.CLASES)[probs.argmax(1)], index=datos["id"])
    salida = sample[["id"]].copy()
    salida["answer"] = salida["id"].map(por_id)
    assert not salida.isna().any().any()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.salida, index=False)
    print("CSV reproducido desde encoder y cabeza guardados:", args.salida)


if __name__ == "__main__":
    main()
