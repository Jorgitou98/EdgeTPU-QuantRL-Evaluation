import rollout_quant_vs_dequant_steps
import os
import subprocess
import csv

num_steps = 500
dir_checkpoints = 'checkpoints_pong_ppo/'
dir_files = os.listdir(dir_checkpoints)
checkpoints = list(filter(lambda name: name.startswith("checkpoint_"), dir_files))
checkpoints = sorted(checkpoints)

csv_results = open(f"{dir_checkpoints}/results/diff_decision_percent.csv", "w")
writer_results = csv.writer(csv_results, delimiter=',')
writer_results.writerow(["Num iters", "Percent diff dec"])
csv_results.close()

csv_results = open(f"{dir_checkpoints}/results/diff_decision_percent.csv", "a", buffering=1)
writer_results = csv.writer(csv_results, delimiter=',')


for checkpoint in checkpoints:
  iters_num = int(checkpoint[-6:])
  diff_dec_perc = rollout_quant_vs_dequant_steps.main(model_dequant = f"checkpoints_pong_ppo/exported_models/checkpoint-{iters_num}.tflite",
                                                      model_quant = f"checkpoints_pong_ppo/exported_models/checkpoint-{iters_num}_quant.tflite",
                                                      steps=num_steps)
  writer_results.writerow([iters_num, diff_dec_perc])

csv_results.close()
