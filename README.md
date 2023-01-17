
# Evaluación del tiempo de inferencia y  análisis del error por cuantización sobre el procesador Edge TPU

## Trabajo de Fin de Máster en Ingeniería Informática de la Universidad Complutense de Madrid

### Autor: Jorge Villarrubia Elvira

## ¿Qué contiene en este repositorio?
Este repositorio contiene los *scripts* utilizados para diversos experimentos del Trabajo de Fin de Máster de Jorge Villarrubia. En la memoria del trabajo se explica con detalle el planteamiento de los experimentos y sus resultados, referenciando algunos *scripts* de este repositorio y explicando algunos detalles de su funcionamiento. El repositorio contiene *scripts* para la evaluación del tiempo de inferencia de redes neuronales de capas densas o de convolución usando uno o varios dispositivos [Edge TPU](https://cloud.google.com/edge-tpu?hl=es), y también para el análisis del error por cuantización de modelos de convolución entrenados mediante aprendizaje por refuerzo. El repositorio también contiene los resultados de los experimentos y los *notebooks* utilizados para representarlos gráficamente tal y como se muestran en la memoria.

## Requerimientos para utilizar los *scripts*

Los *scripts* se han ejecutado utilizando la versión de Python 3.8.12, pero deberían funcionar con versiones >= 3.8. En caso de usar anaconda es muy fácil y recomendable crear un entorno ejecutando `$ conda create -n myEnv python=3.8` y activándolo con `$ conda activate myEnv`.

Además, es necesario instalar diversas librerías que se indican a continuación (se han utilizado en las versiones indicadas aunque muy probablemente también funcionen con versiones posteriores):

 - `pycoral 2.0.0`
 - `tensorflow 2.7.0`
 - `torch 1.10.1`
 - `ray 1.13.0`
 - `gym 0.21.0`

Del paquete Ray se utiliza la biblioteca RLlib y de Gym se utilizan los entornos de juegos Atari. Al instalar Ray y Gym conviene indicar estas dependencias para que se instalen directamente RLlib y dichos entornos de juegos:

    pip install pycoral == 2.0.0
    pip install "ray[rllib] == 1.13.0" "tensorflow == 2.7.0" "torch == 1.10.1"
    pip install "gym[atari] == 0.21.0" "gym[accept-rom-license] == 0.21.0" "atari_py == 0.2.9"
    
Además, es necesario instalar correctamente el Edge TPU en el sistema *host* donde se ejecuten los *scripts* . Existen diversas opciones como un [módulo que se puede conectar por USB](https://coral.ai/products/accelerator) o un [módulo de tipo M.2 que se conecta por un puerto PCIe](https://coral.ai/products/m2-accelerator-ae). Por otra parte, es necesario descargar el compilador de Edge TPU tal y como se explica en [la documentación](https://coral.ai/docs/edgetpu/compiler/) , donde también se explica cómo utilizarlo.

## Uso de *scripts* para los principales experimentos

### Tiempo de inferencia  y uso de memoria en Edge TPU aumentando progresivamente el número de operaciones MAC de modelos de capas convolucionales.

El *script* [`CONV_MACs_memory_test.py`](https://github.com/Jorgitou98/EdgeTPU-QuantRL-Evaluation/blob/main/CONV_MACs_memory_test.py) genera y evalúa sistemáticamente modelos de capas convolucionales en el Edge TPU  según se describe en la memoria del trabajo. Para la generación de modelos se varía el número de filtros de cada capa entre los valores `--minF` y `--maxF` recibidos como parámetro, con un paso `--stepF`. Otros parámetros que puede recibir el *script* son: 

 - `--input-size`: Entero con el tamaño de la entrada del modelo. Realmente la entrada simula una imagen cuadrada con 3 canales y por tanto es input-size $\times$ input-size $\times$ 3. Por defecto su valor es 64.
 - `--steps`: El número de inferencias a ejecutar con cada modelo tomando la media de tiempos de las ejecuciones como resultado. Por defecto vale 100.
 - `--layers`: Número de capas de convolución con los filtros indicados que tiene el modelo. Por defecto vale 5.
 - `--segments-list`: Una lista con los diferentes números de TPUs a evaluar con ejecuciones segmentadas del modelo. Por defecto es solo una TPU.
 - `--batch-size`: Tamaño del lote de entradas con el que se realiza la inferencia del modelo. Tal y como se explica en la memoria es importante en los experimentos con segmentación y varias TPUs. Por defecto vale 1.
 - `--profile-partition`: Un *booleano* para indicar que si se quiere usar perfilado para la segmentación. En caso de que su valor sea falso se usa la segmentación que ofrece el compilador de Edge TPU. Por defecto vale *False*.

A continuación se muestra un ejemplo de ejecución del *script* correspondiente al experimento con perfilado y *batch* de tamaño 50 cuyos resultados se exponen en la memoria del trabajo:

    $ python CONV_MACs_memory_test.py --minF 32 --maxF 702 --stepF 10 --layers 5 --segments 5 --segments-list 1,2,3,4 --batch-size 50 --profile-partition True

El *script* muestra diversos mensajes por pantalla con los tiempos que va midiendo de cada modelo y reportes de compilación. Además, genera un fichero de salida `CONV_MACs/results/minF{minF}-maxF{maxF}-stepF{stepF}-seg{num_segments}-batch{batch_size}-profiling{profile_partition}.csv` con una fila por cada modelo que contiene el número de filtros de cada capa, el número de operaciones MACs, el uso de memoria interna de la TPU en MiB, el uso de memoria externa en MiB (memoria del *host*)  y el tiempo medio de inferencia en milisegundos.

### Tiempo de inferencia  y uso de memoria en Edge TPU aumentando progresivamente el número de operaciones MAC de modelos de capas densas.
Se trata de un experimento análogo al anterior pero requiere una implementación propia para la segmentación ya que el compilador de Edge TPU no funciona correctamente para capas densas (es un error ajeno a nosotros, tal y como se explica en la memoria). Dado que hay bastantes diferenciasn entre el caso sin segmentación, el caso con segmentación por defecto y el caso con segmentación basada en perfilado, se ha implementado un *script*  para cada uno en lugar de tener un solo *script* que abarque los tres a base de poner muchas condiciones todos el rato.

 1. El *script* [`FC_MACs_memory_test.py`](https://github.com/Jorgitou98/EdgeTPU-QuantRL-Evaluation/blob/main/FC_MACs_memory_test.py) permite realizar el experimento en un Edge TPU sin segmentación. Recibe los valores mínimos y máximos de neuronas para los modelos a generar (`--minN` y `--maxN`), el paso para el cambio de neuronas de un modelo al siguiente (`--stepN`) , el número de capas de los modelos (`--layers`) y el número de repeticiones por cada modelo para obtener su tiempo de inferencia (`--steps`). A continuación se muestra cómo se ejecutaría el experimento para los resultados con una sola TPU de capas densas mostrado en la memoria del trabajo: 

	```
	$ python FC_MACs_memory_test.py --minN 100 --maxN 2640 --stepN 40 --layers 5 --steps 50
	```

 2. El *script* [`FC_MACs_memory_test_multisegment.py`](https://github.com/Jorgitou98/EdgeTPU-QuantRL-Evaluation/blob/main/FC_MACs_memory_test_multisegment.py) permite realizar el experimento con varios segmentos siguiendo el esquema por defecto de reparto del modelo (más detalles en la memoria). Su uso es análogo al del anterior *script* pero además recibe el número de segmentos en los que fraccionar el modelo (`--num_segments`) y el tamaño del lote de entradas para las inferencias (`--batch`). A continuación mostramos un ejemplo de cómo ejecutarlo para reproducir el experimento con lote de tamaño 50 y 4 segmentos que se expone en la memoria:
	```
	$ python FC_MACs_memory_test_multisegment.py --minN 100 --maxN 2640 --stepN 40 --batch 50 --num_segments 4 --layers 5 --steps 50
	```
 3. El *script* [`FC_MACs_memory_test_profiling.py`](https://github.com/Jorgitou98/EdgeTPU-QuantRL-Evaluation/blob/main/FC_MACs_memory_test_profiling.py) permite una ejecución del modelo segmentada previo perfilado de las distintas opciones la manera de fraccionar el modelo (ver explicación en la memoria). La manera de ejecutarlo es análoga a la del *script* para segmentación por defecto:
	```
	$ python FC_MACs_memory_test_profiling.py --minN 100 --maxN 2640 --stepN 40 --batch 50 --num_segments 4 --layers 5 --steps 50
	```
