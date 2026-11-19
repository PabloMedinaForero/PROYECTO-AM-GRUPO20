# Análisis de las entradas al modelo

Mejor candidato: `p2_exp13_ultimas4_mlp`. Accuracy: 0.89057; anterior: 0.88837.
Corrige 50 reseñas antes erradas y pierde 41 antes acertadas: +9 aciertos netos de 4.094.

## Comparación de métodos

| Método | Entrada | Representación | Cabeza | Accuracy | Época |
| --- | --- | --- | --- | ---: | ---: |
| p2_exp13_ultimas4_mlp | parrafo | ultimas4 | mlp | 0.89057 | 7 |
| p2_exp10_cls_promedio_lineal | parrafo | cls_promedio | lineal | 0.89008 | 1 |
| p2_exp11_cls_promedio_mlp | parrafo | cls_promedio | mlp | 0.88984 | 7 |
| p2_exp12_mitades_mlp | parrafo | mitades | mlp | 0.88959 | 7 |
| p2_exp08_cls_lineal | parrafo | cls | lineal | 0.88886 | 5 |
| p2_exp09_promedio_lineal | parrafo | promedio | lineal | 0.88813 | 2 |
| p2_exp16_maximo_mlp | parrafo | cls_promedio_maximo | mlp | 0.88740 | 1 |
| p2_exp15_segmentos_mlp | dos_segmentos | mitades | mlp | 0.88349 | 7 |
| p2_exp14_segmentos_lineal | dos_segmentos | mitades | lineal | 0.88251 | 7 |

La combinación CLS + promedio de las últimas cuatro capas obtiene el mejor resultado local. Separar oraciones en dos segmentos no mejora en estas condiciones. El encoder se había ajustado con párrafos; no se hizo un nuevo ajuste de sus pesos con pares.

En las nueve variantes se usa el mismo encoder supervisado del 80%, congelado, y el mismo aumento. Se comparan entradas, representaciones y cabezas; no es una ablación del aumento. La comparación con el modelo anterior también cambia la regularización y la normalización de la cabeza, de modo que no permite atribuir toda la diferencia a las últimas cuatro capas.

## Idioma estimado

| Idioma | n | Anterior | Nuevo |
| --- | ---: | ---: | ---: |
| en | 472 | 0.84534 | 0.84746 |
| es | 2732 | 0.90483 | 0.91105 |
| fr | 419 | 0.84010 | 0.83532 |
| mixto_es_en | 4 | 1.00000 | 0.75000 |
| pt | 467 | 0.87794 | 0.86510 |

El idioma es un diagnóstico estimado. El grupo mixto es demasiado pequeño para sacar conclusiones.

## Ejemplos reservados

### Corregidos

- ID 1395, fr; real: positivo; antes: negativo; ahora: positivo. Texto: Je l'ai achetée à la mi-année et elle m'est arrivée en une dizaine de jours, à peu près. Ce qui m'a déçue, c'est la taille, du début à la fin, franchement je m'attendais à autre chose. Je m'en sers quand même tous les jours pour le téléphone dans la poche du manteau. Je recommande la housse les yeux fermés, sans hésiter.
- ID 1801, es; real: positivo; antes: negativo; ahora: positivo. Texto: Llego el ventilador el martes por la tarde, lo pedi para el cuarto porque en las noches hace calor. Aqui va mi resena honesta: quede muy molesto con el ventilador. El sistema de sonido lo probe conectandolo donde esta el televisor, ahi mismo quedaron instaladas las bocinas viejas que teniamos. De paso, no tengo ni una sola queja del sistema de sonido. No sabria bien que recomendar.
- ID 2084, es; real: positivo; antes: negativo; ahora: positivo. Texto: Compre esta plancha de pelo en marzo para mi hija que se va al colegio temprano. Mi hermana me dijo 'take it back' porque las planchas baratas nunca sirven, pero no la devolvi, y menos mal. Mi mamá la usó un día, la agarró y dijo 'no, this one's actually good'. Yo casi la regalo a mi tía pero al final no la regale, porque la verdad entre todas en la casa la usamos diario. El servicio tecnico ni la vimos, no hizo falta. Pase del caleton al pelo liso en diez minutos, honestly.

### Regresiones

- ID 336, en; real: negativo; antes: negativo; ahora: positivo. Texto: I bought this camera at the beginning of the year because I wanted it for pool outings with my family on weekends, and it arrived in about four days. Here's my take. The water resistance was a total win. On top of that, the image sharpness was a purchase mistake. Judge for yourselves.
- ID 604, fr; real: negativo; antes: negativo; ahora: positivo. Texto: L'aspiration sur le carrelage est franchement puissante, les poils du chat disparaissent en un passage. Mais l'enrouleur du câble se coince à chaque fois et le bac à poussière est pénible à vider sans en mettre partout. Le poids reste correct pour mon dos. Je ne sais toujours pas si je la garde ou si je tente un autre modèle.
- ID 747, en; real: negativo; antes: negativo; ahora: positivo. Texto: I bought this carry-on in June for my weekly trips to my parents' place, and honestly I keep comparing it to the old suitcase I had for years. The old one rolled better than this one, and its handle felt sturdier than what I remember here. My wife even packed it for her cousin's wedding weekend. Between the two, I'd take the old one without thinking twice.

### Errores que permanecen

- ID 6, pt; real: negativo; antes: positivo; ahora: positivo. Texto: Comprei há umas duas semanas pela internet e chegou em quatro dias. Desde então estou testando a fundo quase todos os dias no escritório. A portabilidade deixa muito a desejar, sinceramente. Por outro lado, o aplicativo respondeu como eu esperava quando configurei. Tem de tudo, como se vê, cada um vai tirar suas próprias conclusões.
- ID 45, en; real: positivo; antes: negativo; ahora: negativo. Texto: The suction power on this vacuum genuinely surprised me; it pulled embedded grit from the hallway rug that the old machine always left behind. On the other hand, emptying the canister releases a small cloud of dust no matter how carefully I do it, which is frustrating every single time. The cord tangles easily too. So part of me wants to keep it and part of me wonders if I made the right call. Still deciding how I feel about it.
- ID 104, es; real: negativo; antes: positivo; ahora: positivo. Texto: Me llegó el reloj el martes, dos días después de pedirlo, y lo uso sobre todo para salir a correr por las tardes. La pantalla me decepcionó de lo lindo. Lo llevo también cuando lavo los trastes y en la ducha, y la resistencia al agua funciona de maravilla.

## Alcance

Se seleccionó entre nueve variantes en una única partición de validación. La ganancia es pequeña y no demuestra una mejora en datos nuevos. El mejor score público confirmado sigue siendo 0,87721. Los nuevos CSV tienen score pendiente.

El candidato particion80 conserva exactamente el encoder y la cabeza evaluados. El candidato full usa el encoder anterior entrenado con todo train y vuelve a entrenar la cabeza y su normalización con todo train; no tiene un score interno independiente.

## Promedios de probabilidades

También se probaron tres promedios simples:
- ultimas4_y_anterior: 0.88666.
- ultimas4_y_lineal: 0.88862.
- tres_modelos: 0.88715.

Ninguno superó al candidato individual; no se generan envíos de estas combinaciones. Para repetir esta comprobación opcional se necesita el archivo de probabilidades anterior junto al CSV de validación.

## Resultado público posterior

El grupo confirmó **0,87819** para `p2_exp13_ultimas4_mlp_full.csv`, frente a **0,87721** del modelo anterior. Es el mejor envío público confirmado hasta ahora. La versión `particion80` continúa sin score reportado. Los experimentos quedan en pausa a solicitud del grupo; se conservan los modelos y resultados para retomarlos después.
