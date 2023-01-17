import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_checkpoints', required = True, help='Path to checkpoints directory')
args = parser.parse_args()

num_episodes = 1
dir_checkpoints = args.dir_checkpoints
dir_files = os.listdir(dir_checkpoints)
checkpoints = list(filter(lambda name: name.startswith("checkpoint_"), dir_files))
checkpoints = sorted(checkpoints)

for checkpoint in checkpoints:
  iters_num = int(checkpoint[-6:])
  save_model_comm = f"python transformer_scripts/model_saver.py {dir_checkpoints}/{checkpoint}/checkpoint-{iters_num} {dir_checkpoints}/exported_models/checkpoint-{iters_num}"
  subprocess.Popen(save_model_comm.split()).communicate()
  conv_tflit_comm = f"python transformer_scripts/tflite_converter_pb.py {dir_checkpoints}/exported_models/checkpoint-{iters_num} {dir_checkpoints}/exported_models/checkpoint-{iters_num}.tflite"
  subprocess.Popen(conv_tflit_comm.split()).communicate()
  quant_comm = f"python transformer_scripts/quantizer.py datasets/dataset-84.npy {dir_checkpoints}/exported_models/checkpoint-{iters_num} {dir_checkpoints}/exported_models/checkpoint-{iters_num}_quant.tflite"
  subprocess.Popen(quant_comm.split()).communicate()
  rm_extra_file = f"rm -r {dir_checkpoints}/exported_models/checkpoint-{iters_num}"
  subprocess.Popen(rm_extra_file.split()).communicate()
