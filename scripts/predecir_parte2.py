"""Genera un submission usando exclusivamente los archivos de la carpeta entrega."""
from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=Path("output/parte2/entrega"))
    parser.add_argument("--eval", type=Path, default=Path("data/parte2/eval.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/parte2/sample_submission.csv"))
    parser.add_argument("--salida", type=Path, default=Path("output/parte2/submission_reproducida.csv"))
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.use_deterministic_algorithms(True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    batch = 16 if dispositivo.type == "cuda" else 8
    checkpoint = torch.load(args.modelo / "modelo_final.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("modo_prueba"):
        raise ValueError("Este modelo es de una prueba reducida, no de la ejecución completa.")
    config = checkpoint["config"]
    neuronas = config["neuronas"]
    dropout = config["dropout"]
    red = nn.Sequential(
        nn.Linear(checkpoint["dimension"], neuronas), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(neuronas, neuronas // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(neuronas // 2, len(checkpoint["clases"])),
    ).to(dispositivo)
    red.load_state_dict(checkpoint["pesos"])
    red.eval()
    tokenizador = AutoTokenizer.from_pretrained(args.modelo / "encoder", local_files_only=True)
    encoder = AutoModel.from_pretrained(args.modelo / "encoder", local_files_only=True).to(dispositivo).eval()
    if encoder.config.model_type != "bert" or encoder.config.is_decoder or encoder.config.is_encoder_decoder:
        raise ValueError("El modelo guardado debe ser el encoder BERT del proyecto.")
    evaluar, sample = pd.read_csv(args.eval), pd.read_csv(args.sample)
    if not evaluar["id"].is_unique or not sample["id"].is_unique or set(evaluar["id"]) != set(sample["id"]):
        raise ValueError("Revisar IDs repetidos o diferentes en eval y sample.")
    if evaluar[["id", "text"]].isna().any().any():
        raise ValueError("Hay nulos en eval.")
    media, escala = checkpoint["media"].to(dispositivo), checkpoint["escala"].to(dispositivo)
    predicciones = []
    with torch.no_grad():
        for inicio in range(0, len(evaluar), batch):
            textos = evaluar["text"].iloc[inicio:inicio + batch].tolist()
            entradas = tokenizador(textos, padding=True, truncation=True, max_length=checkpoint["max_length"],
                                   return_special_tokens_mask=True, return_tensors="pt")
            especiales = entradas.pop("special_tokens_mask").to(dispositivo)
            entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
            ocultos = encoder(**entradas).last_hidden_state
            mascara = (entradas["attention_mask"].bool() & ~especiales.bool()).unsqueeze(-1)
            cls = ocultos[:, 0, :]
            promedio = (ocultos * mascara).sum(1) / mascara.sum(1).clamp(min=1)
            tipo = config["representacion"]
            if tipo == "cls_promedio":
                vectores = torch.cat([cls, promedio], dim=1)
            else:
                vectores = cls if tipo == "cls" else promedio
            predicciones.extend(red((vectores - media) / escala).argmax(1).cpu().tolist())
            if inicio % 800 == 0:
                print(f"Inferencia: {min(inicio + batch, len(evaluar))}/{len(evaluar)}", flush=True)
    etiquetas = np.array(checkpoint["clases"])[predicciones]
    por_id = pd.Series(etiquetas, index=evaluar["id"])
    salida = sample[["id"]].copy()
    salida["answer"] = salida["id"].map(por_id)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.salida, index=False)
    print("Archivo guardado:", args.salida)


if __name__ == "__main__":
    main()
