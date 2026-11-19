### Análisis de la ejecución completa del 18 de noviembre de 2026

La validación contiene 4.094 reseñas originales y el mejor candidato, `p2_exp05_lr_aumento`,
acertó 2.939: accuracy de **0,71788**. Usar CLS solo dio 0,70054 y promedio solo dio 0,69370.
En esta partición no se cumplió la idea de que promediar siempre sería mejor que usar CLS.
Concatenar ambas representaciones con aumento dio 0,70396; reducir la tasa de aprendizaje
de 0,001 a 0,0003 elevó ese resultado a 0,71788. Esta es la mejora más clara de la secuencia.

La matriz de confusión muestra que 1.079 de los 1.155 errores son intercambios entre
positivo y negativo: 515 negativos se predicen positivos y 564 positivos se predicen
negativos. La clase neutral tiene 1.079 aciertos de 1.120 casos, por lo que separar las
reseñas descriptivas funciona bastante mejor que resolver la polaridad de las opiniones.
Las curvas muestran sobreajuste: la pérdida de entrenamiento sigue bajando mientras
la de validación vuelve a subir después de la época 6. Por eso se restauran los pesos
de esa época y se usan seis épocas para reentrenar el candidato final.

El par controlado sin/con aumento pasó de 0,69370 a 0,69492: **cinco aciertos adicionales**,
o **+0,12 puntos porcentuales**. En inglés mejoró 2,97 puntos, pero en francés bajó 1,19;
en español y portugués los cambios fueron menores a un cuarto de punto. El efecto no es
uniforme entre idiomas y una sola partición no demuestra que el aumento mejore siempre.
También cambia la cantidad de actualizaciones por época al agregar reseñas, por lo que
esta comparación mide el procedimiento completo de aumento con el mismo límite de épocas.

**Español estimado: 72,95% de accuracy (n=2.732).** En la reseña 1026 la etiqueta es
positiva: primero critica la nitidez de imagen, pero después elogia el motor con una
concesión. El modelo predice negativo. Un vector que resume toda la reseña conserva los
dos tonos y el MLP no siempre identifica cuál domina. La reseña 3636 contiene frases en
inglés, aunque el detector la agrupó como español. Esto muestra que la regla de mezclas
es conservadora y que las cifras por idioma no son una evaluación con idiomas anotados.

**Inglés estimado: 70,97% (n=472).** La reseña 6339 tiene etiqueta negativa, pero el
modelo predice positivo: el elogio al motor convive con una crítica al cargador. En la
11542 la batería dura varios días, la pantalla falla al aire libre y el comprador expresa
indecisión. Su etiqueta es positiva y el modelo predice negativo. La coexistencia de
opiniones opuestas dificulta distinguir la opinión global; no estamos cambiando las
etiquetas del conjunto para resolver esa ambigüedad.

**Francés estimado: 68,74% (n=419).** La reseña 6265 dice que el soporte resolvió el problema
y que no hay ninguna queja: su etiqueta es positiva, pero se predice negativo. Es un
ejemplo concreto de una negación asociada a una palabra de queja. En la 10308 se menciona
un cambio sin problemas y buen sonido, pero finalmente el producto vuelve al servicio
técnico y deja de usarse. La etiqueta es negativa y el modelo predice positivo: no da el
peso necesario a la consecuencia final.

**Portugués estimado: 68,95% (n=467).** La reseña 4907 usa negaciones como «Não é que eu
goste» y «Não diria que o preço é bom». La etiqueta es negativa y se predice positivo.
No basta con reconocer palabras de elogio si están dentro de una negación. La 3351 combina
ventajas de las sartenes con cabos flojos y dudas sobre devolverlas; también se predice
positivo cuando la etiqueta es negativa. Se repite el problema de ponderar elogios frente
al juicio global y las concesiones.

**Mezcla ES/EN estimada: un acierto de cuatro ejemplos.** La reseña 6695 declara
insatisfacción en inglés y describe problemas con las asas, pero el modelo predice positivo.
Es un caso útil para inspeccionar; **cuatro reseñas son insuficientes para generalizar**.
Además, la 3636 demuestra que otras mezclas quedaron dentro del grupo español.

No se recortó ningún texto original de entrenamiento con el límite de 192 tokens. Por eso,
en esta ejecución, los errores observados no se explican por truncamiento de esas reseñas.
El uso de embeddings congelados reduce el costo y permite comparar el MLP fácilmente,
pero no adapta el encoder a las negaciones y contrastes particulares de estos datos.
Los resultados aquí descritos son de validación interna; todavía no son scores de Kaggle.
