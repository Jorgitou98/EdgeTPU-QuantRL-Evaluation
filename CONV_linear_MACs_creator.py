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

W = 64
input_shape = (W, W, 3)

def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)
    return tflite_model

def representative_data_gen():
    for input_value in np.array(np.random.random_sample([100] + list(input_shape)), dtype=np.float32):
        yield [np.array([input_value])]

def quantize(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    open(model_file_prefix + "_quant.tflite", "wb").write(tflite_model_quant)

def memory_use(num_filters, num_MACs, line_init):
  f = open(f"CONV_linear_MACs/compile_info/N{num_filters}-nMACs{num_MACs}_compiler", 'r')
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

parser = argparse.ArgumentParser()
parser.add_argument('--minF', type=int, default=2, help='Minimum number of filters for de experiment')
parser.add_argument('--maxF', type=int, default=2025, help='Maximum number of filters for de experiment')
parser.add_argument('--stepF', type=int, default=100, help='Step number of filters for de experiment')
parser.add_argument('--steps', type=int, default=100, help='Step per inference')
args = parser.parse_args()

minF = args.minF
maxF = args.maxF
stepF = args.stepF
steps = args.steps

csv_results = open(f"CONV_linear_MACs/results/minF{minF}-maxN{maxF}-stepN{stepF}.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num filters", "# MACs", "On chip mem used", "Off chip mem used", "Inference time"])

for num_filters in range(minF, maxF+1, stepF):
    # w^2 * 3*2 es aplicar un filtro sobre un canal. Hay que hacerlo tantas veces como canales de entrada para cada canal de salida
    num_MACs = W * W * 3 * 3 * (3*num_filters + num_filters*num_filters*4)
    print("num_MACs:", num_MACs, "num_filters:", num_filters, "input_shape:", input_shape)
    model = Sequential()
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    print(model.summary())
    #input("continue")

    model_file_prefix = f"CONV_linear_MACs/N{num_filters}-nMACs{num_MACs}"
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()  # Needed before execution!
    output_details = interpreter.get_output_details()[0]  # Model has single output.
    input_details = interpreter.get_input_details()[0]  # Model has single input.
    print("Output:", output_details)
    print("Input:", input_details)

    quantize(model_file_prefix, model)

    orig_stdout = os.dup(sys.stdout.fileno())
    f = open(f"CONV_linear_MACs/compile_info/N{num_filters}-nMACs{num_MACs}_compiler", 'w')
    os.dup2(f.fileno(), sys.stdout.fileno())
    edge_tpu = f'edgetpu_compiler -o CONV_linear_MACs/ {model_file_prefix}_quant.tflite'
    subprocess.Popen(edge_tpu.split()).communicate()
    os.dup2(orig_stdout, sys.stdout.fileno())
    f.close()

    on_chip_mem_MB = memory_use(num_filters, num_MACs, "On-chip memory used for caching model parameters:")
    off_chip_mem_MB = memory_use(num_filters, num_MACs, "Off-chip memory used for streaming uncached model parameters:")

    inf_time = rollout_edge_tpu_basic.execute(model_path=f"{model_file_prefix}_quant_edgetpu.tflite", steps = steps)
    print("Tiempo de inferencia:", inf_time)
    #input("Continuar")

    writer_results.writerow([num_filters, num_MACs, on_chip_mem_MB, off_chip_mem_MB, inf_time])

csv_results.close()
