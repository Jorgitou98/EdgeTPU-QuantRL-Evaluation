import argparse
import subprocess
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
        '-m', '--model', required=True, type=int, choices=[i for i in range(1,9)], help='Integer indicating the model to create. It must be an integer value between 1 and 6. Run python train_model.available_models() to see the description of available models.'
)
parser.add_argument(
        '-d', '--dim', required = True, type=int, help= 'Input dimension'
)
parser.add_argument(
        '-ts', '--tpus-segment', required = False, nargs="+", default="1", type=int, help= 'Input dimension'
)
args = parser.parse_args()

train = f'python train_pong.py --model={args.model} --gpu=gpu0 --workers=1 --save-name=model{args.model} --iters=0'
subprocess.Popen(train.split()).communicate()

name_base = f"exported_models/batch_test/model{args.model}"

export = f'python model_saver.py checkpoints/ppo/model{args.model}/checkpoint_000001/checkpoint-1 {name_base}'
subprocess.Popen(export.split()).communicate()

convert = f'python tflite_converter_pb.py {name_base} {name_base}.tflite'
subprocess.Popen(convert.split()).communicate()

if not os.path.exists(f'datasets/dataset-{args.dim}.npy'):
  create_dataset = f'python datasets/dataset_creator.py {args.dim} datasets/dataset-{args.dim} Pong-v0'
  subprocess.Popen(create_dataset.split()).communicate()

quant8b = f'python quantizer.py datasets/dataset-{args.dim}.npy {name_base} {name_base}-quant8.tflite'
subprocess.Popen(quant8b.split()).communicate()

edge_tpu = f'edgetpu_compiler {name_base}-quant8.tflite'
subprocess.Popen(edge_tpu.split()).communicate()

move_tpu_file = f'mv model{args.model}-quant8_edgetpu.tflite {name_base}-quant8_edgetpu.tflite'
subprocess.Popen(move_tpu_file.split()).communicate()

move_tpu_log = f'mv model{args.model}-quant8_edgetpu.log {name_base}-quant8_edgetpu.log'
subprocess.Popen(move_tpu_log.split()).communicate()

for num_segments in args.tpus_segment:
  if num_segments == 1:
    continue
  edge_tpu_segm = f'edgetpu_compiler --num_segments={num_segments} {name_base}-quant8.tflite'
  subprocess.Popen(edge_tpu_segm.split()).communicate()
  for segment in range(num_segments):
    rm_tflite_segm = f"rm model{args.model}-quant8_segment_{segment}_of_{num_segments}.tflite"
    subprocess.Popen(rm_tflite_segm.split()).communicate()
    move_tpu_segm_file = f'mv model{args.model}-quant8_segment_{segment}_of_{num_segments}_edgetpu.tflite {name_base}-quant8_segment_{segment}_of_{num_segments}_edgetpu.tflite'
    subprocess.Popen(move_tpu_segm_file.split()).communicate()
    move_tpu_segm_log = f'mv model{args.model}-quant8_segment_{segment}_of_{num_segments}_edgetpu.log {name_base}-quant8_segment_{segment}_of_{num_segments}_edgetpu.log'
    subprocess.Popen(move_tpu_segm_log.split()).communicate()

