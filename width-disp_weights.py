import os
import subprocess
import csv
import argparse
import tflite_runtime.interpreter as tflite
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dir_checkpoints', required = True, help='Path to checkpoints directory')
args = parser.parse_args()

dir_checkpoints = args.dir_checkpoints
dir_files = os.listdir(dir_checkpoints)
checkpoints = list(filter(lambda name: name.startswith("checkpoint_"), dir_files))
checkpoints = sorted(checkpoints)

csv_results = open(f"{dir_checkpoints}/results/width-dispersion_weights.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num iters", "weights concentration"])
csv_results.close()

csv_results = open(f"{dir_checkpoints}/results/width-dispersion_weights.csv", "a", buffering=1)
writer_results = csv.writer(csv_results, delimiter=',')

for checkpoint in checkpoints:
  iters_num = int(checkpoint[-6:])
  model_dequant = f"{dir_checkpoints}/exported_models/checkpoint-{iters_num}.tflite"
  tflite_interpreter = tflite.Interpreter(model_path=model_dequant)
  tflite_interpreter.allocate_tensors()

  tensor_details = tflite_interpreter.get_tensor_details()
  weights_means = []
  width_disp = 0
  num_values = 0
  for dict in tensor_details:
    i = dict['index']
    tensor_name = dict['name']
    if tensor_name[-6::] != "Conv2D":
      continue
    tensor = tflite_interpreter.tensor(i)()
    for i in range(tensor.shape[0]):
      width_disp += max(abs(np.min(tensor[0])), np.max(tensor[0]))/np.std(tensor[0]) * math.prod(tensor[0].shape)
      num_values += math.prod(tensor[0].shape)

  width_disp /= num_values
  print(width_disp)

  writer_results.writerow([iters_num, width_disp])

csv_results.close()
