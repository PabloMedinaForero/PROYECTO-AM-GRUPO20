# Análisis de la mejora de la parte 2

Ejecución del 18 de noviembre de 2026. Se conservaron exactamente los 4.094 IDs de validación
de la primera versión. El score público anterior informado por el grupo es **0,70383**.
Las cifras siguientes son de validación interna y no sustituyen un envío a Kaggle.

| Modelo | Accuracy local | Aciertos / 4.094 |
| --- | ---: | ---: |
| mBERT congelado + MLP, candidato anterior | 0,71788 | 2.939 |
| mBERT ajustado + MLP, sin aumento | 0,88495 | 3.623 |
| mBERT ajustado + MLP, con aumento | **0,88837** | **3.637** |

La mejora frente al modelo congelado fue de **17,05 puntos porcentuales**. Se corrigieron
904 errores del modelo anterior y se perdieron 206 aciertos: el balance fue de 698 aciertos
adicionales. Los errores totales bajaron de 1.155 a 457. La mayor mejora proviene del ajuste
de las capas contextuales, no de un gran aumento en el tamaño de la cabeza de clasificación.
La cabeza conserva dos capas ocultas, de 128 y 64 neuronas.

## Efecto del aumento y selección

Con el mismo entrenamiento y arquitectura, el aumento pasó de 0,88495 a 0,88837: 14 aciertos
netos adicionales, equivalentes a **0,34 puntos porcentuales**. El efecto es pequeño frente
al cambio del encoder congelado al ajustado. Solo se aumentaron reseñas de entrenamiento;
no se aumentaron ni entrenaron las de validación durante esta comparación.

Se probaron también mezclas de las probabilidades de ambos candidatos. La mezcla al 50%
obtuvo 0,88691; las mezclas al 25% y al 75% tampoco superaron el modelo aumentado individual.
Se mantuvo por tanto un solo modelo, con mayor accuracy local y menor costo de inferencia.

En el modelo aumentado, la accuracy pasó de 0,88642 en la época 2 a 0,88837 en la época 4.
La pérdida, en cambio, subió de 0,23673 a 0,53087. Esto muestra sobreajuste y exceso de confianza
en algunos errores, aunque se ganaron ocho aciertos. Se seleccionó la época 4 por la métrica
oficial de accuracy. No se interpreta la elevada accuracy de entrenamiento como evidencia
de generalización. La mejora de la época 4 sobre la 2 es pequeña y depende de esta partición.

## Idiomas estimados

| Grupo estimado | Casos | Accuracy del modelo elegido |
| --- | ---: | ---: |
| Inglés | 472 | 84,53% |
| Español | 2.732 | 90,48% |
| Francés | 419 | 84,01% |
| Portugués | 467 | 87,79% |
| Mezcla ES/EN detectada | 4 | 100,00% |

Los cuatro ejemplos del último grupo no representan todas las mezclas del conjunto.
La reseña 3636 mezcla español e inglés y fue agrupada como español. El clasificador recibe
directamente el texto; estas estimaciones se usan para la partición y el diagnóstico.

## Ejemplos que cambiaron y errores restantes

- **Español, ID 1026:** critica la nitidez de imagen y después elogia el motor mediante una
  concesión. Su etiqueta es positiva. El modelo congelado predijo negativo y el ajustado
  predijo positivo.
- **Inglés, ID 6339:** el elogio al motor convive con una crítica al cargador. Su etiqueta es
  negativa. El modelo ajustado corrigió la predicción positiva del modelo anterior.
- **Francés, ID 6265:** el soporte resolvió el problema y el texto declara que no tiene ninguna
  queja. Su etiqueta es positiva. El modelo anterior predijo negativo y el nuevo positivo.
  En el ID 10308, la devolución al servicio técnico y el abandono del uso también se clasifican
  ahora como negativos, pese a los elogios intermedios.
- **Portugués, ID 4907:** contiene negaciones como «Não é que eu goste» y «Não diria que o preço
  é bom». El modelo elegido todavía predice positivo frente a la etiqueta negativa. El ajuste
  sin aumento sí acertaba este caso: una mayor accuracy global no implica corregir todos los
  mismos ejemplos.
- **Inglés, ID 45, y francés, ID 1030:** combinan ventajas, problemas y una conclusión indecisa.
  Persisten errores cuando el sentimiento global es indirecto o ambas polaridades tienen
  bastante presencia. Se conservan las etiquetas originales del conjunto.
- **Español, ID 104, y portugués, ID 6:** una crítica fuerte se combina con una frase posterior
  positiva o moderada. El modelo todavía puede dar demasiado peso a esa parte final.

La matriz de confusión tiene 206 negativos predichos como positivos y 243 positivos predichos
como negativos. La clase neutral registra 1.115 aciertos de 1.120 casos. Distinguir las dos
polaridades sigue siendo más difícil que reconocer reseñas descriptivas.

## Reentrenamiento y límites

Para el candidato completo se usan las 20.466 reseñas originales de train y 10.227 copias
tipográficas. Se vuelve al modelo preentrenado inicial y se repite el entrenamiento por las
cuatro épocas seleccionadas. Este ajuste final incorpora también los registros antes reservados
para validación, por lo que no se utiliza esa misma partición para atribuirle otra accuracy.

El CSV `particion80` corresponde al checkpoint realmente medido sobre la validación reservada.
El CSV `full` corresponde al reentrenamiento con todo train. Los dos provienen de modelos
guardados y recargados. Los resultados públicos nuevos quedan pendientes de Kaggle.

La comparación usa una sola semilla y una sola partición. Los idiomas son estimados, el aumento
puede introducir ruido y la distribución de eval puede ser distinta de train. No se usan los
textos de eval para entrenamiento, adaptación del encoder, aumento ni selección de épocas.
