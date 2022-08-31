from keras.models import Sequential
from keras.layers import Dense
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

W = 48
input_size = 3 * W * W
output_size = 10

def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)
    return tflite_model

def representative_data_gen():
    for input_value in np.array(np.random.random_sample([100,input_size]), dtype=np.float32):
        yield [input_value]

def quantize(model_file_prefix, keras_model):       
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    open(model_file_prefix + "_quant.tflite", "wb").write(tflite_model_quant)

def memory_use(hidden_neurons, num_MACs, line_init):
  f = open(f"FC_linear_MACs/compile_info/N{hidden_neurons}-nMACs{num_MACs}_compiler", 'r')
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
parser.add_argument('--minN', type=int, default=2, help='Minimum number of neurons for de experiment')
parser.add_argument('--maxN', type=int, default=2025, help='Maximum number of neurons for de experiment')
parser.add_argument('--stepN', type=int, default=100, help='Step number of neurons for de experiment')
parser.add_argument('--num-seg', type=int, nargs="+", default=1, help='List of num segements for model pipelines')

args = parser.parse_args()

minN = args.minN
maxN = args.maxN
stepN = args.stepN
num_segments_list = args.num_seg

csv_results = open(f"FC_linear_MACs/results/minN{minN}-maxN{maxN}-stepN{stepN}.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Hidden neurons", "# MACs", "On chip mem used", "Off chip mem used", "Inference time"])

for hidden_neurons in range(minN, maxN+1, stepN):
    num_MACs = hidden_neurons * (input_size + output_size)
    print("num_MACs:", num_MACs, "hidden_neurons:", hidden_neurons, "input_size:", input_size, "output_size:", output_size)
    model = Sequential()
    model.add(Dense(hidden_neurons, input_shape=(input_size,), activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))

    model_file_prefix = f"FC_linear_MACs/N{hidden_neurons}-nMACs{num_MACs}"
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()  # Needed before execution!
    output_details = interpreter.get_output_details()[0]  # Model has single output.
    input_details = interpreter.get_input_details()[0]  # Model has single input.
    print("Output:", output_details)
    print("Input:", input_details)

    quantize(model_file_prefix, model)

    for num_segments in num_segments_list:
      orig_stdout = os.dup(sys.stdout.fileno())
      f = open(f"FC_linear_MACs/compile_info/N{hidden_neurons}-nMACs{num_MACs}_pipeline_compiler", 'w')
      os.dup2(f.fileno(), sys.stdout.fileno())
      print(num_segments)
      input("Continuar")
      edge_tpu = f'edgetpu_compiler --num_segments={num_segments} {model_file_prefix}_quant.tflite'
      subprocess.Popen(edge_tpu.split()).communicate()
      os.dup2(orig_stdout, sys.stdout.fileno())
      f.close()

      input("Continuar")

      if num_segments > 1 and not os.path.exists(f"N{hidden_neurons}-nMACs{num_MACs}_quant_0_of_{num_segments}_edgetpu.tflite"):
        continue

      on_chip_mem_MB = memory_use(hidden_neurons, num_MACs, "On-chip memory used for caching model parameters:")
      off_chip_mem_MB = memory_use(hidden_neurons, num_MACs, "Off-chip memory used for streaming uncached model parameters:")

      move_tpu_file = f'mv N{hidden_neurons}-nMACs{num_MACs}_quant_edgetpu.tflite {model_file_prefix}_quant_edgetpu.tflite'
      subprocess.Popen(move_tpu_file.split()).communicate()

      move_tpu_log = f'mv N{hidden_neurons}-nMACs{num_MACs}_quant_edgetpu.log {model_file_prefix}_quant_edgetpu.log'
      subprocess.Popen(move_tpu_log.split()).communicate()

      inf_time = rollout_edge_tpu_basic.execute(model_path=f"{model_file_prefix}_quant_edgetpu.tflite", steps = 50)
      print("Tiempo de inferencia:", inf_time)
      #input("Continuar")

    writer_results.writerow([hidden_neurons, num_MACs, on_chip_mem_MB, off_chip_mem_MB, inf_time])

csv_results.close()
