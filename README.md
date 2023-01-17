# Evaluación del tiempo de inferencia y  análisis del error por cuantización sobre el procesador Edge TPU

## Trabajo de fin de Máster en Ingeniería Informática

### Universidad Complutense de Madrid

### Jorge Villarrubia Elvira

# ¿Qué contiene en este repositorio?
Este repositorio contiene los *scripts* utilizados para diversos experimentos del Trabajo de Fin de Máster de Jorge Villarrubia. En la memoria del trabajo se explica con detalle el planteamiento de los experimentos y sus resultados, referenciando algunos *scripts* de este repositorio y explicando algunos detalles de su funcionamiento. El repositorio contiene *scripts* para la evaluación del tiempo de inferencia de redes neuronales de capas densas o de convolución usando uno o varios dispositivos [Edge TPU](https://cloud.google.com/edge-tpu?hl=es), y también para el análisis del error por cuantización de modelos de convolución entrenados mediante aprendizaje por refuerzo. El repositorio también contiene los resultados de los experimentos y los *notebooks* utilizados para representarlos gráficamente tal y como se muestran en la memoria.

# Requerimientos para utilizar los *scripts*

Los *scripts* se han ejecutado utilizando la versión de Python 3.8.12, pero deberían funcionar con versiones >= 3.8. Además, es necesario instalar diversas librerías que se indican a continuación (se han utilizado en las versiones indicadas aunque muy probablemente también funcionen con versiones posteriores):

 - `pycoral 2.0.0`
 - `tensorflow 2.7.0`
 - `ray 1.13.0`
 - `gym 0.21.0`

Del paquete Ray se utiliza la biblioteca RLlib y de Gym se utilizan los entornos de juegos Atari. Al instalar Ray y Gym conviene indicar estas dependencias para que se instalen directamente RLlib y dichos entornos de juegos:

    pip install "ray[rllib] == 1.13.0"
    pip install "gym[atari] == 0.21.0" "gym[accept-rom-license] == 0.21.0" atari_py == 0.2.9
    
Además, es necesario instalar correctamente el Edge TPU en el sistema *host* donde se ejecuten los *scripts* . Existen diversas opciones como un [módulo que se puede conectar por USB](https://coral.ai/products/accelerator) o un [módulo de tipo M.2 que se conecta por un puerto PCIe](https://coral.ai/products/m2-accelerator-ae). Por otra parte, es necesario descargar el compilador de Edge TPU tal y como se explica en [la documentación](https://coral.ai/docs/edgetpu/compiler/) , donde también se explica cómo utilizarlo.

# Uso de *scripts* para los principales experimentos

### Tiempo de inferencia  y uso de memoria en Edge TPU aumentando progresivamente el número de operaciones MAC de modelos de capas convolucionales

El *script* [`CONV_MACs_memory_test.py`](https://github.com/Jorgitou98/EdgeTPU-QuantRL-Evaluation/blob/main/CONV_MACs_memory_test.py) genera y evalúa sistemáticamente modelos de capas convolucionales en el Edge TPU  según se describe en la memoria del trabajo. Para la generación de modelos se varía el número de filtros de cada capa entre los valores `--minF` y `--maxF` recibidos como parámetro, con un paso `--stepF`. Otros parámetros que puede recibir el *script* son: 

 - `--steps`: El número de inferencias a ejecutar con cada modelo tomando la media de tiempos de las ejecuciones como resultado. Por defecto vale 100.
 - `--layers`: Número de capas de convolución con los filtros indicados que tiene el modelo. Por defecto vale 5.
 - `--segments-list`: Una lista con los diferentes números de TPUs a evaluar con ejecuciones segmentadas del modelo. Por defecto es solo una TPU.
 - `--batch-size`: Tamaño del lote de entradas con el que se realiza la inferencia del modelo. Tal y como se explica en la memoria es importante en los experimentos con segmentación y varias TPUs. Por defecto vale 1.
 - `--profile-partition`: Un *booleano* para indicar que si se quiere usar perfilado para la segmentación. En caso de que su valor sea falso se usa la segmentación que ofrece el compilador de Edge TPU. Por defecto vale *False*.

A continuación se muestra un ejemplo de ejecución del *script* correspondiente al experimento con perfilado y *batch* de tamaño 50 cuyos resultados se exponen en la memoria del trabajo:

    $ python CONV_MACs_memory_test.py --minF 32 --maxF 702 --stepF 10 --layers 5 --segements 5 --segments-list 1,2,3,4 --batch-size 50 --profile-partition True

El *script* muestra diversos mensajes por pantalla con los tiempos que va midiendo de cada modelo y reportes de compilación. Además, genera un fichero de salida `CONV_MACs/results/minF{minF}-maxF{maxF}-stepF{stepF}-seg{num_segments}-profiling{profile_partition}.csv` con una fila por cada modelo que contiene el número de filtros de cada capa, el número de operaciones MACs, el uso de memoria interna de la TPU en MiB, el uso de memoria externa en MiB (memoria del *host*)  y el tiempo medio de inferencia en milisegundos. 
