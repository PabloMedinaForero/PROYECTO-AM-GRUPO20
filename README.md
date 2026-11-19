# Proyecto de Aprendizaje de Máquina - Grupo 20

La parte 1 se conserva en `proyecto1.ipynb`, `data/`, `submissions/` y `models/modelo_final.joblib`.
La parte 2 está en la rama `codex/proyecto-2`; conserva los notebooks de cada etapa.

## Experimentos de entrada posteriores al score 0,87721

`proyecto2_entradas.ipynb` compara nueve variantes de la entrada y de su representación:
párrafo completo o dos segmentos; CLS, promedio, CLS + promedio, mitades, máximo y promedio
de las últimas cuatro capas; clasificadores lineales o un MLP pequeño. Conserva la puntuación,
los acentos y las negaciones. No se usan traducciones ni modelos generativos.

El mejor candidato local es `p2_exp13_ultimas4_mlp`: **0,89057**, frente al **0,88837** anterior
en los mismos 4.094 IDs (**9 aciertos netos más**). Se congela el encoder que ya fue ajustado
para sentimiento con el 80% de train y se entrenan las nuevas cabezas. El encoder completo
solo se utiliza para preparar el candidato full. Esta etapa no es una comparación sin/con aumento:
todos los candidatos mantienen aumento y se regularizan sus cabezas. La separación en dos
segmentos rindió peor; no se volvió a ajustar todo BERT con esa entrada.

**Mejor envío público confirmado:** `output/parte2/entradas/submissions/p2_exp13_ultimas4_mlp_full.csv`,
con **0,87819**, informado por el grupo. Supera el score anterior de 0,87721 en 0,00098
(0,098 puntos porcentuales). También se conserva `p2_exp13_ultimas4_mlp_particion80.csv`
en la misma carpeta; su score público sigue pendiente. Los experimentos quedan en pausa
a solicitud del grupo mientras otras personas prueban alternativas.

```powershell
.\.venv\Scripts\python.exe -u scripts/entradas_parte2.py
.\.venv\Scripts\python.exe scripts/predecir_entradas_parte2.py
```

Los nueve resultados, curvas, errores y diagnóstico por idioma se guardan en
`output/parte2/entradas/`; la explicación está en `analisis_validacion.md` y en el notebook.
Los clasificadores están en `models/parte2/entradas/`. La inferencia independiente usa
`output/parte2/entradas/entrega/encoder/` y `clasificador.pt`, que incluye la normalización.
No necesita train ni el cache de embeddings. `verificacion.json` registra la reproducción del CSV.

Para repetir el notebook en Kaggle, agregar los CSV oficiales y los dos encoders de la etapa
anterior, conservando `validacion/encoder/` y `completo/encoder/`; se detectan dentro de `/kaggle/input`.
Agregar también `validacion_p2_exp07_mbert_ajustado_aumento.csv` para el análisis de aciertos
pareado, y preferiblemente `particion.csv`. Si hay varias copias de esos archivos, ajustar sus rutas
en la primera celda. Activar GPU. Los checkpoints ya son locales y no requieren descargar otro modelo.
La etapa de ajuste original se reproduce con `proyecto2_mejorado.ipynb`.

## Mejora posterior al score 0,70383

El grupo informó un score público de **0,70383** para `p2_exp05_lr_aumento`.
Ese valor ya está registrado en `output/parte2/registro_kaggle.csv`; no se inventaron fechas ni IDs de envío.

La nueva comparación alcanzó **0,88837** con aumento y **0,88495** sin aumento sobre los mismos
4.094 ejemplos de validación: **+17,05 puntos porcentuales** frente al MLP congelado.
El grupo confirmó **0,87180** para el checkpoint `particion80` y **0,87721** para el reentrenado `full`.
El mejor envío público confirmado es **full: 0,87721**.

Archivos para los próximos envíos:

- Primero: `output/parte2/mejora/p2_exp07_mbert_ajustado_aumento_full.csv`, reentrenado con todo train.
- Alternativa: `output/parte2/mejora/p2_exp07_mbert_ajustado_aumento_particion80.csv`, el checkpoint medido en validación.

Los scores reales de estos candidatos se registran en `output/parte2/mejora/registro_kaggle.csv`.
Al ejecutar de nuevo el notebook, el mejor score público registrado determina qué checkpoint se copia
a la carpeta de entrega. Si todavía no hay scores, la selección permanece provisional.

`proyecto2_mejorado.ipynb` y `scripts/finetuning_parte2.py` implementan el ajuste de los doce
bloques contextuales de mBERT junto con el MLP. Se conservan el vocabulario preentrenado,
la restricción encoder-only y los mismos 4.094 ejemplos de validación. Se comparan dos versiones,
sin y con aumento. Los archivos de la primera versión se conservan para poder comparar.

```powershell
.\.venv\Scripts\python.exe -u scripts/finetuning_parte2.py
```

La nueva ejecución escribe en `output/parte2/mejora/` y guarda pesos en
`models/parte2/finetuning/`. El modelo ganador local se reentrena con todo train y se recarga
para generar el nuevo CSV. Los scores de validación nuevos no son scores de Kaggle.
Para ejecutar en Kaggle, importar el notebook mejorado, agregar los CSV de la competencia y
activar GPU e Internet. Si se agrega también `particion.csv`, reutiliza esos IDs; de lo contrario,
reconstruye la misma división con `langid==1.1.6` y semilla 42.

Para repetir la inferencia mejorada desde los archivos guardados:

```powershell
.\.venv\Scripts\python.exe scripts/predecir_mejora_parte2.py
```

Para Bloque Neón, usar el notebook mejorado ejecutado, `referencias_parte2.pdf` y la carpeta
`output/parte2/mejora/entrega/` correspondiente al candidato finalmente elegido. El análisis de esta
ejecución está en `output/parte2/mejora/analisis_validacion.md`. La entrega incluye `encoder/`,
`cabeza.pt`, configuración y el CSV; todos esos componentes se necesitan para reproducirla.

## Parte 2

Clasificación multilingüe con **mBERT congelado + MLP de dos capas ocultas**, siguiendo la práctica
de embeddings y los conceptos del MLP del curso. Solo se usa un Transformer encoder-only.
El notebook es autónomo: no necesita importar archivos Python del repositorio en Kaggle.

**Ejecución completa verificada el 18 de noviembre de 2026:** los cinco experimentos se entrenaron
con los CSV oficiales en una RTX 3060. El mejor resultado local fue 0,71788 de accuracy
(`p2_exp05_lr_aumento`). Los cinco submissions tienen 6.781 filas y predicciones distintas.
La inferencia independiente desde los archivos de entrega reprodujo las 6.781 predicciones del
CSV final. El grupo confirmó posteriormente un score público de 0,70383 para ese modelo.

Incluye cinco experimentos, comparación controlada con/sin aumento tipográfico, validación
estratificada, errores por idioma estimado, curvas, checkpoints PyTorch y CSV alineados por ID.
El idioma se estima para diagnóstico; el clasificador recibe directamente el texto multilingüe.

### Ejecutar en Kaggle

1. Abrir la [competencia de la parte 2](https://www.kaggle.com/competitions/parte-2-competencia-aprendizaje-de-maquina-2026-20).
2. Crear un notebook privado e importar `proyecto2.ipynb` mediante la opción de importar notebook.
3. Agregar los datos de esta competencia, seleccionar una GPU y activar Internet para descargar mBERT.
4. Ejecutar la celda opcional `%pip install langid==1.1.6` si falta esa dependencia. No reinstalar PyTorch si la GPU ya funciona.
5. Ejecutar todo y guardar una versión con los outputs. La extracción inicial de embeddings es la parte más costosa.
6. Descargar los cinco CSV de `/kaggle/working/output/parte2/submissions/` y enviarlos individualmente desde **Submit Prediction**.
7. Guardar score público, fecha e ID en `output/parte2/registro_kaggle.csv`. En Kaggle se puede editar el registro desde una celda:

```python
registro = pd.read_csv(RUTA_REGISTRO)
# Reemplazar el nombre y el valor por los del envío REAL recibido en Kaggle.
# registro.loc[registro.experimento == "p2_exp03_promedio_aumento", "accuracy_publica"] = SCORE_REAL
registro.to_csv(RUTA_REGISTRO, index=False)
```

8. Volver a ejecutar las secciones 11 y 12. Se copia el checkpoint exacto del mejor envío público, sin reentrenarlo otra vez.
9. Conservar las evidencias de al menos cinco envíos distintos. El notebook avisa si dos CSV tienen exactamente las mismas predicciones.

Si todavía no hay scores públicos, la selección queda marcada como **provisional por validación**.
Generar los archivos no realiza los envíos automáticamente ni acredita haber superado la línea base privada.
No enviar los CSV de `submissions/` en la raíz: pertenecen a la parte 1 y solo tienen 3.000 filas.

### Ejecutar localmente

Se recomiendan Python 3.11 o 3.12 y una GPU compatible; también funciona con CPU.
Los CSV oficiales de la parte 2 están en `data/parte2/` (20.466/6.781 filas).
Para GPU NVIDIA, instalar la variante CUDA indicada por el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/).
Una instalación que termina en `+cpu` no usa la GPU, aunque el equipo tenga una.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-parte2.txt
.\.venv\Scripts\python.exe scripts/ejecutar_proyecto2.py
```

El notebook descarga una revisión fija de mBERT. También acepta una copia local en
`models/parte2/mbert/`; la carpeta debe corresponder a la revisión indicada en el notebook.
Los archivos grandes, el entorno y el cache están excluidos de Git.

Para comprobar el flujo con 300 reseñas y dos épocas:

```powershell
.\.venv\Scripts\python.exe scripts/ejecutar_proyecto2.py --prueba
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La prueba reducida escribe solo en `output/parte2/pruebas/`. Sus métricas y modelos no son resultados
del proyecto completo y sus CSV no sirven para la competencia. La ejecución completa guarda el notebook
ejecutado en `proyecto2.ipynb`; si falla, conserva allí las celdas y el error para diagnosticarlo.

### Entrega y archivos

| Archivo o carpeta | Uso |
| --- | --- |
| `proyecto2.ipynb` | Implementación autónoma y análisis, ejecutar y conservar outputs |
| `referencias_parte2.pdf` | Documento adicional de fuentes, licencias y uso declarado de IA |
| `referencias_parte2.md` | Versión editable de las referencias |
| `output/parte2/resultados_validacion.csv` | Resultados reales de los experimentos |
| `output/parte2/resultados_por_idioma.csv` | Desempeño por idioma estimado y efecto del aumento |
| `output/parte2/analisis_errores.md` | Interpretación de la ejecución completa con ejemplos de validación por idioma |
| `output/parte2/particion.csv` | IDs usados en entrenamiento y validación |
| `output/parte2/aumento_entrenamiento.csv` | Copias aumentadas y sus reseñas de origen |
| `output/parte2/modelos/` | Checkpoint exacto de cada candidato |
| `output/parte2/registro_kaggle.csv` | Scores reales registrados por el grupo |
| `output/parte2/submissions_parte2.zip` | Los cinco CSV preparados para extraer y enviar individualmente |
| `output/parte2/entrega/` | Modelo final, encoder/tokenizador, configuración y submission |

Para Bloque Neón se necesitan el notebook ejecutado, el documento de referencias y **toda** la carpeta
`entrega/`. El `.pt` guarda el MLP y la normalización; `encoder/` guarda mBERT y el tokenizador.
La sección 12 comprueba que los archivos recargados reproduzcan el CSV final.

Para repetir la inferencia sin volver a entrenar ni ejecutar todos los experimentos:

```powershell
.\.venv\Scripts\python.exe scripts/predecir_parte2.py
```

Este script usa únicamente `output/parte2/entrega/`, los textos de evaluación y el sample;
no descarga modelos ni necesita las reseñas de entrenamiento.

La estrategia y las limitaciones están explicadas dentro del notebook. El grupo debe revisar los ejemplos
de errores y comprender el código para la sustentación. El documento de referencias declara la asistencia
de OpenAI Codex utilizada en código y documentación.
