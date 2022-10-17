from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf
import numpy as np
import subprocess
import rollout_cpu_basic
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

parser = argparse.ArgumentParser()
parser.add_argument('--minN', type=int, default=2, help='Minimum number of neurons for de experiment')
parser.add_argument('--maxN', type=int, default=2025, help='Maximum number of neurons for de experiment')
parser.add_argument('--stepN', type=int, default=100, help='Step number of neurons for de experiment')
parser.add_argument('--steps', type=int, default=100, help='Step per inference')
parser.add_argument('--layers', type=int, default=100, help='Number of neural layers')
args = parser.parse_args()

minN = args.minN
maxN = args.maxN
stepN = args.stepN
steps = args.steps
L = args.layers


csv_results = open(f"FC_MACs/results/cpu-minN{minN}-maxN{maxN}-stepN{stepN}.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num neuronas", "# MACs", "Inference time"])

for num_neurons in range(minN, maxN+1, stepN):
    num_MACs = num_neurons * (input_size + output_size + L * num_neurons)
    print("num_MACs:", num_MACs, "num_neurons:", num_neurons, "input_size:", input_size, "output_size:", output_size)
    model = Sequential()
    model.add(layers.Dense(num_neurons, input_shape=(input_size,), activation='tanh', use_bias=True, bias_initializer='zeros'))
    for _ in range(L-1):
      model.add(layers.Dense(num_neurons, activation='tanh', use_bias=True,bias_initializer='zeros'))
    model.add(layers.Dense(output_size, use_bias=True, bias_initializer='zeros'))
    print(model.summary())

    model_file_prefix = f"FC_MACs/N{num_neurons}-nMACs{num_MACs}"
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
    inf_time = rollout_cpu_basic.execute(model_path=f"{model_file_prefix}.tflite", steps = steps)
    writer_results.writerow([num_neurons, num_MACs, inf_time])
csv_results.close()
