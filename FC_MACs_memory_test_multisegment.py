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
import rollout_edge_tpu_pipeline_batch_FC

input_size = 64
output_size = 10
input_segment = 100

def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)
    return tflite_model

def representative_data_gen():
    for input_value in np.array(np.random.random_sample([100,input_size]), dtype=np.float32):
        yield [input_value]

def representative_data_gen_int8():
    for input_value in np.array(np.random.random_sample([100,input_segment]), dtype=np.float32):
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

def memory_use(hidden_neurons, num_MACs, line_init):
  f = open(f"FC_MACs/compile_info/N{hidden_neurons}-nMACs{num_MACs}_compiler", 'r')
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
parser.add_argument('--minN', type=int, default=2, help='Minimum number of neurons for de experiment')
parser.add_argument('--maxN', type=int, default=2025, help='Maximum number of neurons for de experiment')
parser.add_argument('--stepN', type=int, default=100, help='Step number of neurons for de experiment')
parser.add_argument('--layers', type=int, default=1, help='Step number of neurons for de experiment')
parser.add_argument('--num_segments', type=int, default=2, help='Number of segments for TPU pipeline execution')
parser.add_argument('--batch', type=int, default=1, help='Batch size for TPU pipeline execution')
parser.add_argument('--steps', type=int, default=50, help='Repetitions for execution')

args = parser.parse_args()

minN = args.minN
maxN = args.maxN
stepN = args.stepN
L = args.layers
num_segments = args.num_segments
batch = args.batch
steps = args.steps

csv_results = open(f"FC_MACs/results/minN{minN}-maxN{maxN}-stepN{stepN}-L{L}-num_seg{num_segments}-batch{batch}.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Hidden neurons", "# MACs", "On chip mem used", "Off chip mem used", "Inference time"])
csv_results.close()

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

    #Manually split. Number of layer in each layer
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

    #input("continuar")


    on_chip_mem_MB = []
    off_chip_mem_MB = []
    for num_segment in range(num_segments):
      model_file_prefix = f"FC_MACs/N{hidden_neurons}-nMACs{num_MACs}_seg{num_segment}of{num_segments}"
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
      f = open(f"FC_MACs/compile_info/N{hidden_neurons}-nMACs{num_MACs}_compiler", 'w')
      os.dup2(f.fileno(), sys.stdout.fileno())
      edge_tpu = f'edgetpu_compiler -o FC_MACs {model_file_prefix}_quant.tflite'
      subprocess.Popen(edge_tpu.split()).communicate()
      os.dup2(orig_stdout, sys.stdout.fileno())
      f.close()

      on_chip_mem_MB.append(memory_use(hidden_neurons, num_MACs, "On-chip memory used for caching model parameters:"))
      off_chip_mem_MB.append(memory_use(hidden_neurons, num_MACs, "Off-chip memory used for streaming uncached model parameters:"))

    inf_time = rollout_edge_tpu_pipeline_batch_FC.execute(model_prefix=f"FC_MACs/N{hidden_neurons}-nMACs{num_MACs}", num_segments = num_segments, steps = steps, batch = batch)
    print("Tiempo de inferencia:", inf_time)
    #input("Continuar")

    writer_results.writerow([hidden_neurons, num_MACs, on_chip_mem_MB, off_chip_mem_MB, inf_time])
    print("Mem on chip", on_chip_mem_MB)
    print("Mem off chip", off_chip_mem_MB)
    csv_results.close()
