from keras.models import Sequential
from tensorflow.keras import layers, activations
from contextlib import contextmanager
import tensorflow as tf
import numpy as np
import subprocess
import rollout_edge_tpu_basic
import rollout_edge_tpu_pipeline_basic
import rollout_pipeline_batch_CONV
import os
import sys
import re
import csv
import argparse

# Se parsean los argumentos del entrada del programa
parser = argparse.ArgumentParser()
parser.add_argument('--minF', type=int, required = True, help='Minimum number of filters for the experiment')
parser.add_argument('--maxF', type=int, required = True, help='Maximum number of filters for the experiment')
parser.add_argument('--stepF', type=int, default=100, help='Step number of filters for the experiment')
parser.add_argument('--input-size', type=int, default = 64, help='Size for the square image to be received by the model')
parser.add_argument('--steps', type=int, default=100, help='Step per inference')
parser.add_argument('--layers', type=int, default=5, help='Number of convolutional layers.')
parser.add_argument('--segments-list', nargs='+', type=int, default=1, help='List with number of segments for test.')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for execution execution.')
parser.add_argument('--profile-partition', type=bool, default=False, help='Flag for pipeline segmentation using profiling')
args = parser.parse_args()


# Tamaño de las entradas (imágenes cuadradas del tamaño indicado con 3 canales de entrada)
input_shape = (args.input_size, args.input_size, 3)


# Renombramos los argumentos para referirnos a ellos de forma más breve en el resto del código
minF = args.minF
maxF = args.maxF
stepF = args.stepF
steps = args.steps
L = args.layers
segments_list = args.segments_list
batch_size = args.batch_size
profile_partition = args.profile_partition

# Función que convierte un modelo keras a TFLite y lo guarda en un fichero con el prefijo que recibe por parámetro (también lo devuelve)
def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)
    return tflite_model


# Función que genera datos representativos para la cuantización. Como aquí no se evalúa la calidad del modelo basta generar datos aletaorios (da igual la calidad de la cuantización)
def representative_data_gen():
    for input_value in np.array(np.random.random_sample([100] + list(input_shape)), dtype=np.float32):
        yield [np.array([input_value])]


# Función que cuantiza un modelo keras que recibe como parámetro y lo guarda en un fichero con el prefijo que recibe seguido de "_quant.tflite"
def quantize(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    open(model_file_prefix + "_quant.tflite", "wb").write(tflite_model_quant)


# Función que dado el string con el que empieza cierta línea de reporte sobre de memoria y el número de segmentos del reporte devuelve una lista con el uso de memoria que recoge el reporte para cada uno de ellos en MiB.
def memory_use(line_init, num_segments):
  f = open(f"CONV_MACs/compile_info/compiler_aux", 'r')
  lines_mem_used = [line for line in f.readlines() if line_init in line]
  f.close()
  memory_uses= []
  for line_mem_used in lines_mem_used:
    data_parsed = re.search(f"{line_init} (.+?)(B|KiB|MiB)", line_mem_used)
    mem_use = float(data_parsed.group(1))
    mem_magnitude = data_parsed.group(2)
    if mem_magnitude == "KiB":
      mem_use /= 1024
    elif mem_magnitude == "B":
      mem_use /= (1024 * 1024)
    memory_uses.append(mem_use)
  return memory_uses


# Función que dado el prefijo del modelo y el número de segementos ejecuta el modelo en los Edge TPUs correspondientes y devuelve el tiempo de inferenia y usos de memoria obtenidos. Tiene en cuenta si está habilitado el flag de perfilado para ejecutar con la herramienta de perfilado de Coral
def test_edge_tpu(model_file_prefix, num_segments):
    orig_stdout = os.dup(sys.stdout.fileno())
    f = open(f"CONV_MACs/compile_info/compiler_aux", 'w')
    os.dup2(f.fileno(), sys.stdout.fileno())
    edge_tpu = f'edgetpu_compiler --num_segments {num_segments} -o CONV_MACs/ {model_file_prefix}_quant.tflite'
    if profile_partition:
      edge_tpu = f'./libcoral/out/k8/tools/partitioner/partition_with_profiling --edgetpu_compiler_binary /usr/bin/edgetpu_compiler --model_path {model_file_prefix}_quant.tflite --num_segments {num_segments} --diff_threshold_ns {profiling_diff_threshold} --output_dir CONV_MACs/'
    subprocess.Popen(edge_tpu.split()).communicate()
    os.dup2(orig_stdout, sys.stdout.fileno())
    f.close()
    if num_segments > 1 and not os.path.exists(f"{model_file_prefix}_quant_segment_0_of_{num_segments}_edgetpu.tflite"):
      return (None, None, None)
    on_chip_mems_MB = memory_use(line_init="On-chip memory used for caching model parameters:", num_segments=num_segments)[-num_segments:]
    off_chip_mems_MB = memory_use(line_init="Off-chip memory used for streaming uncached model parameters:", num_segments=num_segments)[-num_segments:]
    if num_segments == 1:
      inf_time = rollout_edge_tpu_basic.execute(model_path=f"{model_file_prefix}_quant_edgetpu.tflite", steps = steps)
    else:
      inf_time = rollout_edge_tpu_pipeline_basic.execute(model_prefix=f"{model_file_prefix}_quant", num_segments = num_segments, steps = steps)
    inf_time = rollout_pipeline_batch_CONV.execute(model_prefix=f"{model_file_prefix}_quant", num_segments = num_segments, steps = steps, batch_size = batch_size)
    print(f"Inference time {num_segments} segments:", inf_time)
    return (inf_time, on_chip_mems_MB, off_chip_mems_MB)


# Por cada experimentos con un número de segementos diferentes que haya que ejecutar creamos un fichero .csv con la cabecera para sus resultados
for num_segments in segments_list:
  csv_results = open(f"CONV_MACs/results/minF{minF}-maxF{maxF}-stepF{stepF}-seg{num_segments}-batch{batch_size}-profiling{profile_partition}.csv", "w")
  writer_results = csv.writer(csv_results, delimiter=',')
  writer_results.writerow(["Num filters", "# MACs", "On chip mem used", "Off chip mem used", "Inference time"])
  csv_results.close()


# Por cada cantidad de filtros con la que haya que generar un modelo
for num_filters in range(minF, maxF+1, stepF):
    # Calculamos el número de operaciones MAC del modelo
    # w^2 * 3*2 es aplicar un filtro sobre un canal. Hay que hacerlo tantas veces como canales de entrada para cada canal de salida. El +1 es por los bias.
    num_MACs = W * W * 3 * 3 * ((3+1)*num_filters + (num_filters+1)*num_filters*(L-1))
    print("num_MACs:", num_MACs, "num_filters:", num_filters, "input_shape:", input_shape)
    # Creamos el modelo con tantas capas como hayamos recibido, con los filtros de esta iteración
    model = Sequential()
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', input_shape=input_shape, activation='relu'))
    for _ in range(L-1):
      model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    print(model.summary())
    # Establecemos el prefijo del modelo para los ficheros que se iran generando y lo convertimos a formato TFLite, preparándolo para alojar tensores del tamaño de entrada.
    model_file_prefix = f"CONV_MACs/N{num_filters}-nMACs{num_MACs}"
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()[0]
    input_details = interpreter.get_input_details()[0]
    print("Output:", output_details)
    print("Input:", input_details)

    # Cuantizamos el modelo
    quantize(model_file_prefix, model)

    # Por cada cantidad de segmentos diferente con la que haya que evaluar el modelo
    for num_segments in segments_list:
      # Realizamos el experimento con el correspondiente número de segmentos obteniendo el tiempo de inferencia y los usos de memoria
      inf_time, on_chip_mem_MB, off_chip_mem_MB = test_edge_tpu(model_file_prefix=model_file_prefix, num_segments=num_segments)
      print(inf_time, on_chip_mem_MB, off_chip_mem_MB)
      # Añadimos en el correspondiente fichero del experimento los resultados de este modelo
      csv_results = open(f"CONV_MACs/results/minF{minF}-maxF{maxF}-stepF{stepF}-seg{num_segments}-batch{batch_size}-profiling{profile_partition}.csv", "a")
      writer_results = csv.writer(csv_results, delimiter=',')
      writer_results.writerow([num_filters, num_MACs, on_chip_mem_MB, off_chip_mem_MB, inf_time])
csv_results.close()
