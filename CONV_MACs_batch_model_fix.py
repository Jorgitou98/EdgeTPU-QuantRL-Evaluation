from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf
import numpy as np
import subprocess
import rollout_edge_tpu_basic
import rollout_edge_tpu_pipeline_basic
import rollout_edge_tpu_pipeline_batch
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

parser = argparse.ArgumentParser()
parser.add_argument('--minB', type=int, default=1, help='Minimum size of batch for de experiment')
parser.add_argument('--maxB', type=int, default=100, help='Maximum size of batch for de experiment')
parser.add_argument('--stepB', type=int, default=5, help='Step batch for de experiment')
parser.add_argument('--steps', type=int, default=500, help='Step per inference')
parser.add_argument('--segments-list', nargs='+', type=int, default=1, help='List with number of segments for test.')
parser.add_argument('--profile-partition', type=bool, default=False, help='Flag for pipeline segmentation using profiling')
parser.add_argument('--profiling-diff-threshold', type=int, default=500000, help='Threshold between the slowest and fastest segment in ns')
args = parser.parse_args()

minB = args.minB
maxB = args.maxB
stepB = args.stepB
steps = args.steps
segments_list = args.segments_list
profile_partition = args.profile_partition
profiling_diff_threshold = args.profiling_diff_threshold


for num_segments in segments_list:
  csv_results = open(f"CONV_MACs/results/minB{minB}-maxB{maxB}-stepB{stepB}-seg{num_segments}-profiling{profile_partition}.csv", "w")
  writer_results = csv.writer(csv_results, delimiter=',')
  writer_results.writerow(["Batch size", "Inference time"])
  csv_results.close()

num_filters = 672
model = Sequential()
model.add(layers.Conv2D(num_filters, (3,3), padding='same', input_shape=input_shape, activation='relu'))
model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
print(model.summary())

model_file_prefix = f"CONV_MACs/N{num_filters}"
tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()  # Needed before execution!
output_details = interpreter.get_output_details()[0]  # Model has single output.
input_details = interpreter.get_input_details()[0]  # Model has single input.
print("Output:", output_details)
print("Input:", input_details)

quantize(model_file_prefix, model)

for num_segments in segments_list:
  edge_tpu = f'edgetpu_compiler --num_segments {num_segments} -o CONV_MACs/ {model_file_prefix}_quant.tflite'
  if profile_partition:
    edge_tpu = f'./libcoral/out/k8/tools/partitioner/partition_with_profiling --edgetpu_compiler_binary /usr/bin/edgetpu_compiler --model_path {model_file_prefix}_quant.tflite --num_segments {num_segments} --diff_threshold_ns {profiling_diff_threshold} --output_dir CONV_MACs/'
  subprocess.Popen(edge_tpu.split()).communicate()

  for batch_size in range(minB, maxB+1, stepB):
    inf_time = rollout_edge_tpu_pipeline_batch.execute(model_prefix=f"{model_file_prefix}_quant", num_segments = num_segments, steps = steps, batch_size = batch_size)
    csv_results = open(f"CONV_MACs/results/minB{minB}-maxB{maxB}-stepB{stepB}-seg{num_segments}-profiling{profile_partition}.csv", "a")
    writer_results = csv.writer(csv_results, delimiter=',')
    writer_results.writerow([batch_size, inf_time])
