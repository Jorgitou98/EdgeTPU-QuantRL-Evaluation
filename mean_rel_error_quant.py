import rollout_quant_vs_dequant_steps
import os
import subprocess
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_checkpoints', required = True, help='Path to checkpoints directory')
args = parser.parse_args()

num_steps = 10000
dir_checkpoints = args.dir_checkpoints
dir_files = os.listdir(dir_checkpoints)
checkpoints = list(filter(lambda name: name.startswith("checkpoint_"), dir_files))
checkpoints = sorted(checkpoints)

csv_results = open(f"{dir_checkpoints}/results/quant_error_{num_steps}steps.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num iters", "Mean rel error", "Diff actions"])
csv_results.close()

csv_results = open(f"{dir_checkpoints}/results/quant_error_{num_steps}steps.csv", "a", buffering=1)
writer_results = csv.writer(csv_results, delimiter=',')

for checkpoint in checkpoints:
  iters_num = int(checkpoint[-6:])
  model_dequant = f"{dir_checkpoints}/exported_models/checkpoint-{iters_num}.tflite"
  model_quant = f"{dir_checkpoints}/exported_models/checkpoint-{iters_num}_quant.tflite"
  mean_eror_quant, diff_act = rollout_quant_vs_dequant_steps.main(model_quant = model_quant, model_dequant = model_dequant, env_name="Pong-v0", steps = num_steps)
  writer_results.writerow([iters_num, mean_eror_quant, diff_act])

csv_results.close()
