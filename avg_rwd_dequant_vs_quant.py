import rollout_tflite_rwd_avg
import os
import subprocess
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_checkpoints', required = True, help='Path to checkpoints directory')
args = parser.parse_args()

num_episodes = 500
dir_checkpoints = args.dir_checkpoints
dir_files = os.listdir(dir_checkpoints)
checkpoints = list(filter(lambda name: name.startswith("checkpoint_"), dir_files))
checkpoints = sorted(checkpoints)

csv_results_dequant = open(f"{dir_checkpoints}/results/results_dequant_250ep.csv", "w")
writer_results_dequant = csv.writer(csv_results_dequant, delimiter=',')
writer_results_dequant.writerow(["Num iters", "Rwd avg"])
csv_results_dequant.close()

csv_results_quant = open(f"{dir_checkpoints}/results/results_quant_250ep.csv", "w")
writer_results_quant = csv.writer(csv_results_quant, delimiter=',')
writer_results_quant.writerow(["Num iters", "Rwd avg"])
csv_results_quant.close()

csv_results_dequant = open(f"{dir_checkpoints}/results/results_dequant_250ep.csv", "a", buffering=1)
writer_results_dequant = csv.writer(csv_results_dequant, delimiter=',')
csv_results_quant = open(f"{dir_checkpoints}/results/results_quant_250ep.csv", "a", buffering=1)
writer_results_quant = csv.writer(csv_results_quant, delimiter=',')

for checkpoint in checkpoints:
  iters_num = int(checkpoint[-6:])
  rwd_avg_dequant = rollout_tflite_rwd_avg.main(model = f"{dir_checkpoints}/exported_models/checkpoint-{iters_num}.tflite", episodes=num_episodes)
  writer_results_dequant.writerow([iters_num, rwd_avg_dequant])
  rwd_avg_quant = rollout_tflite_rwd_avg.main(model = f"{dir_checkpoints}/exported_models/checkpoint-{iters_num}_quant.tflite", episodes=num_episodes)
  writer_results_quant.writerow([iters_num, rwd_avg_quant])

csv_results_quant.close()
csv_results_dequant.close()
