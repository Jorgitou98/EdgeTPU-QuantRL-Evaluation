from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf
import numpy as np
import subprocess
import rollout_edge_tpu_basic
import os
import sys
from contextlib import contextmanager
import re
import csv
import argparse

# Se parsean los argumentos del entrada del programa
parser = argparse.ArgumentParser()
parser.add_argument('--minN', type=int, default=2, help='Minimum number of neurons for de experiment')
parser.add_argument('--maxN', type=int, default=2025, help='Maximum number of neurons for de experiment')
parser.add_argument('--stepN', type=int, default=100, help='Step number of neurons for de experiment')
parser.add_argument('--layers', type=int, default=1, help='Step number of neurons for de experiment')
parser.add_argument('--input-size', type=int, default = 64, help='Size for the square image to be received by the model')
parser.add_argument('--steps', type=int, default=100, help='Step per inference')
parser.add_argument('--segments-list', nargs='+', type=int, default=1, help='List with number of segments for test.')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for execution execution.')
parser.add_argument('--profile-partition', type=bool, default=False, help='Flag for pipeline segmentation using profiling')
args = parser.parse_args()


# Renombramos los argumentos para referirnos a ellos de forma más breve en el resto del código
minN = args.minN
maxN = args.maxN
stepN = args.stepN
steps = args.steps
input_size = args.input_size
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
    for input_value in np.array(np.random.random_sample([100,input_size]), dtype=np.float32):
        yield [input_value]

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
def memory_use(hidden_neurons, num_MACs, line_init):
  f = open(f"FC_MACs/compile_info/N{hidden_neurons}-nMACs{num_MACs}_compiler", 'r')
  line_mem_used = [line for line in f.readlines() if line_init in line][0]
  f.close()
  data_parsed = re.search(f"{line_init} (.+?)(B|KiB|MiB)", line_mem_used)
  mem = data_parsed.group(1)
  mem_magnitude = data_parsed.group(2)
  print(f"{line_init} {mem}{mem_magnitude}")
  mem_MB = float(mem)
  if mem_magnitude == "KiB":
    mem_MB /= 1024
  elif mem_magnitude == "B":
    mem_MB /= (1024 * 1024)
  return mem_MB


def default_segmentation(num_segments, model):
    layers_per_segment = int((L+1)/num_segments)
    layers_per_segment_list = [layers_per_segment for segment in range(num_segments)]
    for i in range((L+1) % num_segments):
      layers_per_segment_list[-i-1] += 1
    model_segments = []
    last_layer_given = 0
    for lay in layers_per_segment_list:
      segment = Model(model.get_layer(f"dense_layer{last_layer_given+1}").input, model.get_layer(f"dense_layer{last_layer_given+lay}").output)
      print(segment.summary())
      model_segments.append(segment)
      last_layer_given = last_layer_given + lay
    return model_segments
    



csv_results = open(f"FC_MACs/results/minN{minN}-maxN{maxN}-stepN{stepN}-L{L}-I{input_size}.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Hidden neurons", "# MACs", "On chip mem used", "Off chip mem used", "Inference time"])

for hidden_neurons in range(minN, maxN+1, stepN):
    csv_results = open(f"FC_MACs/results/minN{minN}-maxN{maxN}-stepN{stepN}-L{L}-num_seg{num_segments}-batch{batch}.csv", "a")
    writer_results = csv.writer(csv_results, delimiter=',')

    num_MACs = hidden_neurons * (input_size + output_size + (L-1) * hidden_neurons)
    print("num_MACs:", num_MACs, "hidden_neurons:", hidden_neurons, "input_size:", input_size, "output_size:", output_size)
    model = Sequential()
    model.add(layers.Dense(hidden_neurons, input_shape=(input_size,), activation='tanh', use_bias=True, bias_initializer='zeros', name = "dense_layer1"))
    for i in range(L-1):
      model.add(layers.Dense(hidden_neurons, activation='tanh', use_bias=True, bias_initializer='zeros', name = f"dense_layer{i+2}"))
    model.add(layers.Dense(output_size, use_bias=True, bias_initializer='zeros', name = f"dense_layer{L+1}"))
    print(model.summary())

    model_file_prefix = f"FC_MACs/N{hidden_neurons}-nMACs{num_MACs}"
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()[0]
    input_details = interpreter.get_input_details()[0]
    print("Output:", output_details)
    print("Input:", input_details)

    quantize(model_file_prefix, model)

    orig_stdout = os.dup(sys.stdout.fileno())
    f = open(f"FC_MACs/compile_info/N{hidden_neurons}-nMACs{num_MACs}_compiler", 'w')
    os.dup2(f.fileno(), sys.stdout.fileno())
    edge_tpu = f'edgetpu_compiler -o FC_MACs {model_file_prefix}_quant.tflite'
    subprocess.Popen(edge_tpu.split()).communicate()
    os.dup2(orig_stdout, sys.stdout.fileno())
    f.close()

    on_chip_mem_MB = memory_use(hidden_neurons, num_MACs, "On-chip memory used for caching model parameters:")
    off_chip_mem_MB = memory_use(hidden_neurons, num_MACs, "Off-chip memory used for streaming uncached model parameters:")

    #move_tpu_file = f'mv N{hidden_neurons}-nMACs{num_MACs}_quant_edgetpu.tflite {model_file_prefix}_quant_edgetpu.tflite'
    #subprocess.Popen(move_tpu_file.split()).communicate()

    #move_tpu_log = f'mv N{hidden_neurons}-nMACs{num_MACs}_quant_edgetpu.log {model_file_prefix}_quant_edgetpu.log'
    #subprocess.Popen(move_tpu_log.split()).communicate()

    inf_time = rollout_edge_tpu_basic.execute(model_path=f"{model_file_prefix}_quant_edgetpu.tflite", steps = 500)
    print("Tiempo de inferencia:", inf_time)
    #input("Continuar")

    writer_results.writerow([hidden_neurons, num_MACs, on_chip_mem_MB, off_chip_mem_MB, inf_time])

csv_results.close()
