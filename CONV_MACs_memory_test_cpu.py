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

W = 64
input_shape = (W, W, 3)

def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)
    return tflite_model

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

csv_results = open(f"CONV_MACs/results/cpu-minF{minF}-maxF{maxF}-stepF{stepF}.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num filters", "# MACs", "Inference time"])

for num_filters in range(minF, maxF+1, stepF):
    # w^2 * 3*2 es aplicar un filtro sobre un canal. Hay que hacerlo tantas veces como canales de entrada para cada canal de salida. El +1 es por los bias
    num_MACs = W * W * 3 * 3 * ((3+1)*num_filters + (num_filters+1)*num_filters*4)
    print("num_MACs:", num_MACs, "num_filters:", num_filters, "input_shape:", input_shape)
    model = Sequential()
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    print(model.summary())

    model_file_prefix = f"CONV_MACs/N{num_filters}-nMACs{num_MACs}"
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)
    
    inf_time = rollout_cpu_basic.execute(model_path=f"{model_file_prefix}.tflite", steps = steps)
    
    writer_results.writerow([num_filters, num_MACs, inf_time])
csv_results.close()
