from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import entradas_parte2 as m


class EntradasTests(unittest.TestCase):
    def test_dos_segmentos_conservan_palabras_y_orden(self):
        for texto in ["No funciona. La pantalla es buena. Lo devolví.",
                      "I would not say it is bad", "Court.", "", "Bom! Mas não compro de novo."]:
            a, b = m.dividir_resena(texto)
            self.assertEqual(a.split() + b.split(), texto.split())

    def test_resumen_excluye_padding_y_especiales(self):
        t = m.torch
        ocultos = t.tensor([[[100., 100.], [2., 4.], [4., 6.], [200., 200.], [999., 999.]]])
        mask = t.tensor([[1, 1, 1, 1, 0]])
        especiales = t.tensor([[1, 0, 0, 1, 1]])
        seg = t.tensor([[0, 0, 1, 1, 0]])
        for entrada in ["parrafo", "dos_segmentos"]:
            r = m.resumir(ocultos, ocultos, mask, especiales, seg, entrada)
            self.assertTrue(t.equal(r["promedio"], t.tensor([[3., 5.]])))
            self.assertTrue(t.equal(r["maximo"], t.tensor([[4., 6.]])))
            self.assertTrue(t.equal(r["primera"], t.tensor([[2., 4.]])))
            self.assertTrue(t.equal(r["segunda"], t.tensor([[4., 6.]])))
        r = m.resumir(ocultos, ocultos, mask, t.ones_like(mask), seg, "parrafo")
        self.assertTrue(t.equal(r["promedio"], t.zeros((1, 2))))
        self.assertTrue(t.equal(r["maximo"], t.zeros((1, 2))))

    def test_rechaza_encoder_completo_para_validacion(self):
        original = m.ENCODER_80
        try:
            m.ENCODER_80 = Path("modelos/completo/encoder")
            with self.assertRaisesRegex(AssertionError, "80%"):
                m.comparar_entradas()
        finally:
            m.ENCODER_80 = original


if __name__ == "__main__":
    unittest.main()
