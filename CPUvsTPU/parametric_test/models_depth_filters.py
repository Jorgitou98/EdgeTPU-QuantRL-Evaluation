import argparse
import subprocess
import os
import csv
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
        '-ds', '--depthstep', required=True, type=int, help='Step for increment depth'
)
parser.add_argument(
        '-mind', '--mindepth', required=True, type=int, help='Integer indicating the min depth'
)
parser.add_argument(
        '-maxd', '--maxdepth', required=True, type=int, help='Integer indicating the max depth'
)
parser.add_argument(
        '-fs', '--filtersstep', required=True, type=int, help='Step for incrementing the number of filters'
)
parser.add_argument(
        '-minf', '--minfilters', required=True, type=int, help='Integer indicating the min number of filterss'
)
parser.add_argument(
        '-maxf', '--maxfilters', required=True, type=int, help='Integer indicating the max number of filters'
)
parser.add_argument(
        '-i', '--iters', required = True, type=int, help= 'Number of training iters to run'
)
parser.add_argument(
        '-d', '--dim', required = True, type=int, help= 'Input dimension'
)
args = parser.parse_args()

with open(f'total_params/total_params[{args.mindepth}-{args.maxdepth}]-[{args.minfilters}-{args.maxfilters}].csv', 'w', newline='') as outcsv:
  writer = csv.writer(outcsv)
  writer.writerow(["DEPTH", "filters", "TOTAL_PARAMS", "CONFIG"])

for depth in range(args.mindepth, args.maxdepth+1, args.depthstep):
  for filters in range(args.minfilters, args.maxfilters+1, args.filtersstep):

    train = f'python ../training_scripts/train_pong_parametric.py --depth={depth} --filters={filters} --gpu=gpu0 --workers=1 --save-name=model_depth_filters{depth}-{filters} --iters={args.iters}'
    process = subprocess.Popen(train.split())
    output, error = process.communicate()

    checkpoint_number = '0'*(6-len(str(args.iters)))+str(args.iters)
    export = f'python ../savers/model_saver_pong.py ../checkpoints/ppo/model_depth_filters{depth}-{filters}/checkpoint_{checkpoint_number}/checkpoint-{args.iters} ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}'
    process = subprocess.Popen(export.split())
    output, error = process.communicate()

    convert = f'python ../converters/tflite_converter_pong.py ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters} ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}.tflite'
    process = subprocess.Popen(convert.split())
    output, error = process.communicate()

    if not os.path.exists(f'datasets/dataset-model_pong_{args.dim}.npy'):
      create_dataset = f'python datasets/dataset_creator_pong.py {args.dim} datasets/dataset-model_pong_{args.dim}'
      process = subprocess.Popen(create_dataset.split())
      output, error = process.communicate()


    quant8b = f'python ../quantizers/quantizer8b_pong.py datasets/dataset-model_pong_{args.dim}.npy ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters} ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}-quant8.tflite'
    process = subprocess.Popen(quant8b.split())
    output, error = process.communicate()


    edge_tpu = f'edgetpu_compiler ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}-quant8.tflite'
    process = subprocess.Popen(edge_tpu.split())
    output, error = process.communicate()

    move_tpu_files = f'mv checkpoint-{args.iters}-model{depth}-{filters}-quant8_edgetpu.tflite ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}-quant8_edgetpu.tflite'
    process = subprocess.Popen(move_tpu_files.split())
    output, error = process.communicate()

    clear_log = f'rm checkpoint-{args.iters}-model{depth}-{filters}-quant8_edgetpu.log'
    process = subprocess.Popen(clear_log.split())
    output, error = process.communicate()
