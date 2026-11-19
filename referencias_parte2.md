# Referencias y uso de IA - Proyecto de Aprendizaje de Máquina, parte 2

Grupo 20. Universidad de los Andes, semestre 2026-20. Consulta: 18 de noviembre de 2026.

## Datos de la competencia

**Autor / responsable:** Rubén Manrique y equipo del curso. **Fuente:** [Parte 2 - Competencia Aprendizaje de Máquina 2026-20](https://www.kaggle.com/competitions/parte-2-competencia-aprendizaje-de-maquina-2026-20/data). **Licencia publicada en la página de datos:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Se descargaron `train.csv` (20.466 reseñas etiquetadas), `eval.csv` (6.781 reseñas sin etiqueta) y `sample_submission.csv`. No se agregaron datasets externos. Los datos de la primera parte ya están incluidos en el nuevo entrenamiento; no se concatenan nuevamente. Los archivos originales no se modifican. La división interna y el aumento se documentan por separado. Deben conservarse la atribución y las condiciones de la licencia cuando se comparta una adaptación.

## Modelo preentrenado

**Modelo:** `google-bert/bert-base-multilingual-cased` (mBERT). **Autor:** Google / BERT. **Repositorio y descarga:** [Hugging Face](https://huggingface.co/google-bert/bert-base-multilingual-cased). **Licencia declarada:** Apache License 2.0. **Revisión fijada:** `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`.

Se usa solamente el encoder mediante `AutoModel` para construir embeddings contextuales y un MLP de dos capas ocultas para clasificarlos. En la primera versión los pesos del encoder quedaron congelados. En la mejora posterior se entrenan los doce bloques contextuales y el MLP, manteniendo congelada la tabla inicial de embeddings de palabras. Se compara el ajuste con y sin aumento sobre los mismos IDs de validación. No se utilizan una cabeza de generación, un decoder ni una API de etiquetado. El entrenamiento previo de mBERT incluyó Wikipedia multilingüe; ese corpus no se descarga ni se incorpora como dataset adicional del proyecto.

**Referencia académica:** Devlin, J., Chang, M.-W., Lee, K. y Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT. [Artículo](https://aclanthology.org/N19-1423/). La licencia del código original puede consultarse en [google-research/bert](https://github.com/google-research/bert/blob/master/LICENSE).

En la comparación posterior de entradas se congela el encoder ya ajustado para sentimiento y se entrenan cabezas lineales o MLP pequeñas. Se prueban párrafos, pares de segmentos y resúmenes CLS, promedio, máximo, mitades y últimas cuatro capas. Para comparar se usa el encoder del 80%, con la misma validación reservada; el encoder completo solo interviene en el reentrenamiento final. No se agrega otro modelo preentrenado. La mejora interna es pequeña y los nuevos scores de Kaggle quedan pendientes.

## Identificación de idioma

**Herramienta y modelo incluido:** `langid.py`, versión 1.1.6. **Autores:** Marco Lui y Timothy Baldwin. **Fuente y descarga:** [repositorio oficial](https://github.com/saffsd/langid.py) y [PyPI](https://pypi.org/project/langid/). **Licencia:** [BSD de 2 cláusulas](https://github.com/saffsd/langid.py/blob/master/LICENSE).

Se restringe la detección a español, inglés, portugués y francés. Se añade una regla explícita para marcar posibles mezclas español-inglés. Las etiquetas son estimaciones para estratificar y estudiar errores, no anotaciones verificadas ni entradas del clasificador de sentimiento. No se ajusta este detector con `eval.csv`. Si se redistribuye su código o modelo, debe conservarse su aviso de copyright y licencia.

## Datos aumentados

Se toma una muestra del 50% de las reseñas de entrenamiento y se intercambian dos letras interiores de una sola palabra de siete o más letras. Se protegen negaciones y se conservan las etiquetas. Se descartan los cambios vacíos y las copias idénticas. Cada copia conserva el ID de su reseña de origen en un archivo de auditoría. Esta transformación puede introducir ruido: su efecto se mide frente al mismo MLP sin aumento, con la misma validación. No se generan traducciones ni paráfrasis con modelos de lenguaje. Durante los experimentos, el aumento se hace después de separar la validación y exclusivamente sobre la partición de entrenamiento.

## Material del curso utilizado

- `202610/Prácticas/Embeddings/Practica15.ipynb`: BERT multilingüe, representaciones CLS/promedio y clasificación sobre embeddings.
- `202620/Prácticas/P14_Redes neuronales/Perceptrón_Multicapa.ipynb`: capas densas, ReLU, Dropout, Adam, parada temprana y curvas de aprendizaje. La práctica usa Keras y el proyecto expresa estos conceptos con PyTorch.
- `202620/Prácticas/P3_Preparación de textos/P3_Procesamiento_de_Textos.ipynb`: revisión de textos. Se conserva la puntuación y las negaciones para no eliminar información de sentimiento.
- Enunciado `proyecto202620-Parte 2.pdf` y aclaración del profesor sobre el uso exclusivo de Transformers encoder-only.

Los notebooks se consultaron como referencia de conceptos y estilo. No se publica una copia de ese material ni se atribuye una licencia que no esté indicada por sus autores.

## Uso de inteligencia artificial generativa

Se utilizó OpenAI Codex para leer y contrastar el enunciado con el material del curso, proponer e implementar el flujo de entrenamiento, redactar las explicaciones, preparar la exportación y comprobar el código. Esta asistencia incluye código y documentación del proyecto y se declara explícitamente.

No se usó un modelo generativo para asignar sentimientos a los textos de entrenamiento, validación o evaluación, ni para construir las reseñas aumentadas. Las predicciones deben provenir del encoder y el MLP guardados. Las métricas se calculan al ejecutar; los scores de Kaggle se registran únicamente después de recibirlos de la plataforma. El grupo debe revisar y comprender la implementación, interpretar los errores y actualizar esta declaración si realiza otros usos de IA.
