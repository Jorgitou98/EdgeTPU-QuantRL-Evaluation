import os
import subprocess

num_episodes = 1
dir_checkpoints = 'checkpoints_pong_ppo/'
dir_files = os.listdir(dir_checkpoints)
checkpoints = list(filter(lambda name: name.startswith("checkpoint_"), dir_files))
checkpoints = sorted(checkpoints)

for checkpoint in checkpoints:
  iters_num = int(checkpoint[-6:])
  save_model_comm = f"python model_saver.py {dir_checkpoints}/{checkpoint}/checkpoint-{iters_num} checkpoints_pong_ppo/exported_models/checkpoint-{iters_num}"
  subprocess.Popen(save_model_comm.split()).communicate()
  conv_tflit_comm = f"python tflite_converter_pb.py checkpoints_pong_ppo/exported_models/checkpoint-{iters_num} checkpoints_pong_ppo/exported_models/checkpoint-{iters_num}.tflite"
  subprocess.Popen(conv_tflit_comm.split()).communicate()
  quant_comm = f"python quantizer.py datasets/dataset-84.npy checkpoints_pong_ppo/exported_models/checkpoint-{iters_num} checkpoints_pong_ppo/exported_models/checkpoint-{iters_num}_quant.tflite"
  subprocess.Popen(quant_comm.split()).communicate()
  rm_extra_file = f"rm -r checkpoints_pong_ppo/exported_models/checkpoint-{iters_num}"
  subprocess.Popen(rm_extra_file.split()).communicate()
