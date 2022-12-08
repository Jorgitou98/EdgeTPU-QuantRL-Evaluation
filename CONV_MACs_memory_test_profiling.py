from keras.models import Sequential
from tensorflow.keras import layers, activations, Model
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
import rollout_pipeline_batch_FC
import rollout_edge_tpu_basic
import itertools

W = 64
input_shape = (W, W, 3)
channels = 32

def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)
    return tflite_model

def representative_data_gen():
    for input_value in np.array(np.random.random_sample([100] + list(input_shape)), dtype=np.float32):
        yield [input_value]

def representative_data_gen_int8():
    for input_value in np.array(np.random.random_sample([100] + list((W, W,channels))), dtype=np.float32):
        yield [input_value]

def quantize(model_file_prefix, keras_model, num_segment):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen
    if num_segment > 0:
      converter.inference_input_type = tf.int8
      converter.representative_dataset = representative_data_gen_int8

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    open(model_file_prefix + "_quant.tflite", "wb").write(tflite_model_quant)

def memory_use(filters, num_MACs, line_init):
  f = open(f"CONV_MACs/compile_info/F{filters}-nMACs{num_MACs}_compiler", 'r')
  line_mem_used = [line for line in f.readlines() if line_init in line][0]
  f.close()
  data_parsed = re.search(f"{line_init} (.+?)(B|KiB|MiB)", line_mem_used)
  mem = data_parsed.group(1)
  mem_magnitude = data_parsed.group(2)
  #print(f"{line_init} {mem}{mem_magnitude}")
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
parser.add_argument('--layers', type=int, default=1, help='Step number of filters for de experiment')
parser.add_argument('--num_segments', type=int, default=2, help='Number of segments for TPU pipeline execution')
parser.add_argument('--batch', type=int, default=1, help='Batch size for TPU pipeline execution')
parser.add_argument('--steps', type=int, default=50, help='Repetitions for execution')
args = parser.parse_args()

minF = args.minF
maxF = args.maxF
stepF = args.stepF
L = args.layers
num_segments = args.num_segments
batch = args.batch
steps = args.steps

csv_results = open(f"CONV_MACs/results/minF{minF}-maxN{maxF}-stepN{stepF}-L{L}-num_seg{num_segments}-batch{batch}-profiling.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num filters", "# MACs", "On chip mem used", "Off chip mem used", "Inference time"])
csv_results.close()

for filters in range(minF, maxF+1, stepF):
    csv_results = open(f"CONV_MACs/results/minF{minF}-maxN{maxF}-stepN{stepF}-L{L}-num_seg{num_segments}-batch{batch}-profiling.csv", "a")
    writer_results = csv.writer(csv_results, delimiter=',')

    num_MACs = W * W * 3 * 3 * ((3+1)*filters + (filters+1)*filters*(L-1))
    print("num_MACs:", num_MACs, "num_filters:", num_filters)
    model = Sequential()
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', input_shape=input_shape, activation='relu', name = "CONV_layer1"))
    for i in range(L-1):
      model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu', name = f"CONV_layer{i+1}"))
    channels = filters
    print(model.summary())
    # Profiling
    possible_splits = list(itertools.combinations(range(1,L), num_segments-1))
    #best = possible_splits[0]
    inf_times = []
    for split in possible_splits:
      split = [0] + list(split) + [L]
      print("split", split)
      model_segments = []
      for num_segment in range(num_segments):
        model_segments.append(Model(model.get_layer(f"CONV_layer{split[num_segment]+1}").input, model.get_layer(f"CONV_layer{split[num_segment+1]}").output))
      #input("continuar")
      for num_segment in range(num_segments):
        model_file_prefix = f"CONV_MACs/F{filters}-nMACs{num_MACs}_seg{num_segment}_of_{num_segments}"
        tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model_segments[num_segment])
        quantize(model_file_prefix = model_file_prefix, keras_model = model_segments[num_segment], num_segment = num_segment)
        edge_tpu = f'edgetpu_compiler -o FC_MACs {model_file_prefix}_quant.tflite'
        subprocess.Popen(edge_tpu.split()).communicate()
      inf_time = rollout_pipeline_batch_FC.execute(model_prefix=f"CONV_MACs/F{filters}-nMACs{num_MACs}", num_segments = num_segments, steps = 1, batch = 50)
      inf_times.append(inf_time)
    print(inf_times)
    #input("continuar")
    best_split = [0] + list(possible_splits[inf_times.index(min(inf_times))]) + [L+1]
    print(best_split)
    #input("continuar")

    model_segments = []
    for num_segment in range(num_segments):
      model_segments.append(Model(model.get_layer(f"CONV_layer{best_split[num_segment]+1}").input, model.get_layer(f"CONV_layer{best_split[num_segment+1]}").output))
    #input("continuar")


    on_chip_mem_MB = []
    off_chip_mem_MB = []
    for num_segment in range(num_segments):
      model_file_prefix = f"CONV_MACs/F{filters}-nMACs{num_MACs}_seg{num_segment}_of_{num_segments}"
      tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model_segments[num_segment])
      #interpreter = tf.lite.Interpreter(model_content=tflite_model)
      #interpreter.allocate_tensors()  # Needed before execution!
      #output_details = interpreter.get_output_details()[0]  # Model has single output.
      #input_details = interpreter.get_input_details()[0]  # Model has single input.
      #print("Output:", output_details)
      #print("Input:", input_details)
      input_segment = hidden_neurons
      quantize(model_file_prefix = model_file_prefix, keras_model = model_segments[num_segment], num_segment = num_segment)

      orig_stdout = os.dup(sys.stdout.fileno())
      f = open(f"CONV_MACs/compile_info/F{filters}-nMACs{num_MACs}_compiler", 'w')
      os.dup2(f.fileno(), sys.stdout.fileno())
      edge_tpu = f'edgetpu_compiler -o CONV_MACs {model_file_prefix}_quant.tflite'
      subprocess.Popen(edge_tpu.split()).communicate()
      os.dup2(orig_stdout, sys.stdout.fileno())
      f.close()

      on_chip_mem_MB.append(memory_use(hidden_neurons, num_MACs, "On-chip memory used for caching model parameters:"))
      off_chip_mem_MB.append(memory_use(hidden_neurons, num_MACs, "Off-chip memory used for streaming uncached model parameters:"))

    inf_time = rollout_pipeline_batch_FC.execute(model_prefix=f"CONV_MACs/F{filters}-nMACs{num_MACs}", num_segments = num_segments, steps = steps, batch = batch)
    #inf_time = rollout_edge_tpu_pipeline_batch.execute(model_prefix=f"FC_MACs/N{hidden_neurons}-nMACs{num_MACs}", num_segments = num_segments, steps= steps, batch_size= batch)

    print("Tiempo de inferencia:", inf_time)
    #input("Continuar")

    writer_results.writerow([hidden_neurons, num_MACs, on_chip_mem_MB, off_chip_mem_MB, inf_time])
    print("Mem on chip", on_chip_mem_MB)
    print("Mem off chip", off_chip_mem_MB)
    csv_results.close()
