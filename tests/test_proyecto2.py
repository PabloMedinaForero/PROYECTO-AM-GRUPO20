"""Comprobaciones de separación de datos, pooling y exportación del notebook."""
from pathlib import Path
import json
import os
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(RAIZ / "tmp/matplotlib")


class Proyecto2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Leer las funciones del propio notebook: las pruebas usan el código entregado.
        nb = json.loads((RAIZ / "proyecto2.ipynb").read_text(encoding="utf-8"))
        cls.codigo = {"RANDOM_STATE": 42, "FRACCION_AUMENTO": 0.5, "MAX_LENGTH": 192,
                      "EPOCAS": 5, "PACIENCIA": 4, "BATCH_MLP": 16, "BATCH_EMBEDDINGS": 8,
                      "MODELO_BASE": "google-bert/bert-base-multilingual-cased",
                      "REVISION": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
                      "MODO_PRUEBA": True, "CLASES": ["negativo", "neutral", "positivo"]}
        for celda in nb["cells"]:
            if set(celda["metadata"].get("tags", [])) & {"imports", "funciones"}:
                exec(compile("".join(celda["source"]), "proyecto2.ipynb", "exec"), cls.codigo)
        cls.codigo["DISPOSITIVO"] = cls.codigo["torch"].device("cpu")
        cls.codigo["torch"].set_num_threads(2)
        cls.pd, cls.np = cls.codigo["pd"], cls.codigo["np"]

    def test_csv_parte2_y_rechazo_parte1(self):
        leer = self.pd.read_csv
        train, evaluar, sample = [leer(RAIZ / "data/parte2" / n) for n in
                                 ["train.csv", "eval.csv", "sample_submission.csv"]]
        self.codigo["validar_datos"](train, evaluar, sample)
        with self.assertRaisesRegex(ValueError, "PARTE 2"):
            self.codigo["validar_datos"](leer(RAIZ / "data/train.csv"), evaluar, sample)

    def test_submission_alinea_por_id_y_rechaza_errores(self):
        evaluar = self.pd.DataFrame({"id": [8, 2, 19], "text": ["a", "b", "c"]})
        sample = self.pd.DataFrame({"id": [19, 8, 2], "answer": ["neutral"] * 3})
        crear = self.codigo["crear_submission"]
        salida = crear(evaluar, sample, ["positivo", "neutral", "negativo"])
        self.assertEqual(list(salida.columns), ["id", "answer"])
        self.assertEqual(salida["answer"].tolist(), ["negativo", "positivo", "neutral"])
        with self.assertRaises(ValueError):
            crear(evaluar, sample, ["positivo"])
        with self.assertRaises(ValueError):
            crear(evaluar, sample, ["positive", "neutral", "negativo"])
        with self.assertRaises(ValueError):
            crear(evaluar, sample.assign(id=[19, 8, 42]), ["positivo", "neutral", "negativo"])

    def test_aumento_reproducible_sin_contaminar_validacion(self):
        datos = self.pd.read_csv(RAIZ / "data/parte2/train.csv").groupby("label").head(30).copy()
        datos = datos.reset_index(drop=True)
        datos["idioma_estimado"] = "es"
        antes = datos.copy(deep=True)
        it, iv = self.codigo["separar_validacion"](datos)
        aumentado = self.codigo["aumentar_datos"](datos.iloc[it])
        repetido = self.codigo["aumentar_datos"](datos.iloc[it])
        self.pd.testing.assert_frame_equal(aumentado, repetido)
        self.pd.testing.assert_frame_equal(datos, antes)
        self.assertGreater(len(aumentado), 0)
        self.assertTrue(set(aumentado["id_origen"]).isdisjoint(set(datos.iloc[iv]["id"])))
        self.assertTrue(set(aumentado["text"]).isdisjoint(set(datos.iloc[iv]["text"])))
        etiquetas = datos.set_index("id")["label"]
        for _, fila in aumentado.iterrows():
            self.assertEqual(fila["label"], etiquetas.loc[fila["id_origen"]])
            self.assertEqual(len(fila["text"]), len(fila["original"]))
            self.assertEqual(sum(a != b for a, b in zip(fila["text"], fila["original"])), 2)

    def test_aumento_protege_negacion(self):
        texto = "tampoco ninguna without personne jamás nunca não not pas"
        salida = self.codigo["error_tipografico"](texto, self.codigo["random"].Random(42))
        self.assertEqual(texto, salida)
        sin_candidatos = self.pd.DataFrame({"id": [0], "text": ["no va"], "label": ["negativo"]})
        aumentado = self.codigo["aumentar_datos"](sin_candidatos, fraccion=1)
        self.assertTrue(aumentado.empty)
        self.assertEqual(list(aumentado.columns), ["id_origen", "original", "text", "label"])

    def test_pooling_ignora_padding_y_tokens_especiales(self):
        torch = self.codigo["torch"]
        ocultos = torch.tensor([[[100., 100.], [2., 4.], [4., 6.], [200., 200.], [999., 999.]]])
        cls, promedio = self.codigo["resumir_tokens"](
            ocultos, torch.tensor([[1, 1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 1, 1]]))
        self.assertTrue(torch.equal(cls, torch.tensor([[100., 100.]])))
        self.assertTrue(torch.equal(promedio, torch.tensor([[3., 5.]])))
        _, vacio = self.codigo["resumir_tokens"](ocultos, torch.ones((1, 5)), torch.ones((1, 5)))
        self.assertTrue(torch.isfinite(vacio).all())

    def test_entrenamiento_y_checkpoint_reproducen_predicciones(self):
        rng = self.np.random.default_rng(42)
        y = self.np.repeat(self.np.arange(3), 30)
        X = self.np.eye(3)[y] * 4 + rng.normal(0, 0.1, (90, 3))
        escala = self.codigo["StandardScaler"]().fit(X)
        config = {"nombre": "prueba", "neuronas": 16, "dropout": 0., "lr": .02,
                  "representacion": "promedio", "aumento": False}
        norm = escala.transform(X).astype(self.np.float32)
        red, historia, _ = self.codigo["entrenar_mlp"](norm, y, config, epocas=5)
        self.assertLess(historia["loss"].iloc[-1], historia["loss"].iloc[0])
        pred = self.codigo["predecir_vectores"](red, norm)
        self.assertGreater((pred == y).mean(), 0.95)
        with tempfile.TemporaryDirectory(dir=RAIZ / "tmp") as carpeta:
            ruta = Path(carpeta) / "modelo.pt"
            self.codigo["guardar_checkpoint"](ruta, red, escala, config, 5)
            cargada, meta = self.codigo["cargar_checkpoint"](ruta)
            resultado = self.codigo["predecir_checkpoint"](cargada, meta, {"promedio": X, "cls": X})
        self.assertEqual(resultado.tolist(), [self.codigo["CLASES"][i] for i in pred])


if __name__ == "__main__":
    unittest.main()
