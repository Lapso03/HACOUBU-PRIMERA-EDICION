# HaCoBuPrimeraEdicion

## Objetivo

Analizar los datos recogidos por el juego, modificando la dificultad de diferentes parámetros de este:

- Ratio de aparición de los objetos en el juego.
- Distancia al jugador de los objetos.
- Tamaño de los objetos.
- Ratio de recompensa de los objetos.

La dificultad podrá variar según los siguientes valores:

- Tiempo que ha durado la ronda.
- Puntuación conseguida total.
- Puntuación del objeto que menos puntuación le ha dado.
- Puntuación del objeto que más puntuación le ha dado.
- Número de objetos capturados.
- Número de objetos perdidos.
- Tiempo de respuesta medio.
- Tiempo de respuesta mínimo.
- Tiempo de respuesta máximo.

* La intención del juego será mantener al jugador interesado en el mismo, no dificultando el juego hasta valores que sean imposibles de conseguir y realizando un juego demasiado simple que no reporte satisfacción al usuario.
* Para simplificar la implementación y evitar problemas de compatibilidad, se proporcionará a los desarrolladores un conjunto de datos base que coincide con las estructuras generadas por el videojuego y que permita la comunicación del modelo con los videojuegos.
* Además, se proporcionará un servidor FastAPI implementado en Python que permitirá conectarse con los juegos y al que los equipos tendrán que añadir su inferencia. Por consiguiente, el objetivo será generar un modelo de la dificultad del juego y generar el data set de variables como motor de IA.

## Explicación de variables

### Variables generadas por el juego:

  - Tiempo que ha durado la ronda.
    - Medido como un float, en segundos.
  - Puntuación conseguida total.
    - Suma de las puntuaciones que le han dado cada uno de los objetos.
    - Medido como un entero sin límites inferiores o superiores.
  - Puntuación del objeto que menos puntuación le ha dado.
    - Podría ser negativa si hay objetos que resten. Por ejemplo, pescar algas en un juego de pesca.
    - Las puntuaciones de los objetos se comunicarán entre la IA y el juego como floats normalizados, entre -1 y 1. -1 representará la puntuación más baja que un objeto puede dar y 1 la más alta.
  - Puntuación del objeto que más puntuación le ha dado.
    - Las puntuaciones de los objetos se comunicarán entre la IA y el juego como floats normalizados, entre -1 y 1. -1 representará la puntuación más baja que un objeto puede dar y 1 la más alta.
  - Número total de objetos capturados.
    - Medido como un entero positivo.
  - Número total de objetos perdidos.
    - Medido como un entero positivo.
  - Tiempo de respuesta medio.
    - Medido como un float.
    - Tiempo medio que ha tardado un jugador en capturar un objeto desde que aparece.
    - No incluye los objetos que se le han escapado.
  - Tiempo de respuesta mínimo.
    - Medido como un float.
    - Tiempo mínimo que ha tardado un jugador en capturar un objeto desde que ha aparecido.
  - Tiempo de respuesta máximo.
    - Medido como un float.
    - Tiempo máximo que ha tardado un jugador en capturar un objeto desde que ha aparecido.
    - No incluye los objetos que se le han escapado.
   
### Variables generadas por el modelo de IA:
  
- Ratio de aparición de los objetos en el juego.
    - Medido como un float entre 0 y 1. 0 representará la velocidad de aparición más lenta, siendo más **fácil** capturar los objetos, y 1 representará la más rápida.
- Distancia al jugador de los objetos.
    - Medido como un float entre 0 y 1. 0 representará la distancia más cercana al jugador, siendo más **fácil** capturarlo, y 1 la más lejana.
    - Esta variable debería afectar a la generación aleatoria, no aparecer todos a exactamente la misma distancia.
- Tamaño de los objetos.
    - Medido como un float entre 0 y 1. 0 representará los objetos más pequeños, siendo más **difícil** capturarlos, y 1 los más grandes.
- Ratio de recompensa de los objetos.
    - Medido como un float entre -1 y 1. Representará la proporción de objetos que aparecen con diferentes puntuaciones. -1 implicará que aparecerán más objetos con puntuaciones más bajas (o negativas), siendo más **difícil**. 1 representará objetos con puntuaciones más altas, siendo más **fácil**.


## Requerimientos

- Requiere tener Python instalado.
- El archivo videojuego.py contiene una implementación de una API que puede comunicarse con el videojuego.
- La API se puede ejecutar usando el archivo ejecutar.bat.
- El archivo Hackaton.postman_collection.json contiene una colección de PostMan para probar la API.

``Estos archivos solo se usan si quieres intentar jugar y crear tus propios datos para que la IA adapte tu rendimiento, para la ejecución del código no hace faltan los archivos.``
