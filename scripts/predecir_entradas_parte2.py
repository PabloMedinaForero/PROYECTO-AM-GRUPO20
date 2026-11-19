"""Reproduce el candidato de entradas desde sus archivos guardados, sin entrenar."""
from pathlib import Path
import argparse
import json

import pandas as pd
import torch

import entradas_parte2 as p2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=Path("output/parte2/entradas/entrega"))
    parser.add_argument("--eval", type=Path, default=Path("data/parte2/eval.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/parte2/sample_submission.csv"))
    parser.add_argument("--salida", type=Path, default=Path("output/parte2/entradas/submission_reproducida.csv"))
    args = parser.parse_args()
    carpeta_encoder = args.modelo / "encoder"
    meta = torch.load(args.modelo / "clasificador.pt", map_location="cpu", weights_only=True)
    assert meta["encoder_sha256"] == p2.firma_encoder(carpeta_encoder)
    assert meta["clases"] == p2.CLASES and meta["max_length"] == p2.MAX_LENGTH
    encoder = p2.AutoModel.from_pretrained(carpeta_encoder, local_files_only=True).to(p2.DISPOSITIVO)
    assert encoder.config.model_type == "bert" and not encoder.config.is_decoder
    tokenizador = p2.AutoTokenizer.from_pretrained(carpeta_encoder, local_files_only=True)
    datos, sample = pd.read_csv(args.eval), pd.read_csv(args.sample)
    assert datos.id.is_unique and sample.id.is_unique and set(datos.id) == set(sample.id)
    # Se recalculan todos los embeddings; esta comprobación no usa el cache del entrenamiento.
    vectores = p2.extraer(datos.text, encoder, tokenizador, meta["config"]["entrada"])
    probs = p2.predecir_guardado(args.modelo / "clasificador.pt", vectores)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    salida = p2.escribir_submission(probs, datos, sample, args.salida)
    original = pd.read_csv(args.modelo / "submission.csv")
    assert salida.equals(original), "La inferencia no reproduce el CSV guardado."
    verificacion = {"filas": len(salida), "predicciones_identicas": len(salida),
                    "columnas": list(salida.columns), "encoder_sha256": meta["encoder_sha256"],
                    "clases": salida.answer.value_counts().to_dict(), "recalculado_sin_cache": True}
    (args.salida.parent / "verificacion.json").write_text(json.dumps(verificacion, indent=2), encoding="utf-8")
    print("Inferencia independiente verificada:", len(salida), "predicciones idénticas.")


if __name__ == "__main__":
    main()
