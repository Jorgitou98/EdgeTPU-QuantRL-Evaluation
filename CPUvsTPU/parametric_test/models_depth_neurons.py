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
        '-ns', '--neuronsstep', required=True, type=int, help='Step for incrementing neurons'
)
parser.add_argument(
        '-minn', '--minneurons', required=True, type=int, help='Integer indicating the min number of neurons'
)

parser.add_argument(
        '-maxn', '--maxneurons', required=True, type=int, help='Integer indicating the max number of neurons'
)
parser.add_argument(
        '-i', '--iters', required = True, type=int, help= 'Number of training iters to run'
)
parser.add_argument(
        '-d', '--dim', required = True, type=int, help= 'Input dimension'
)
args = parser.parse_args()

with open(f'total_params/total_params[{args.mindepth}-{args.maxdepth}]-[{args.minneurons}-{args.maxneurons}].csv', 'w', newline='') as outcsv:
  writer = csv.writer(outcsv)
  writer.writerow(["DEPTH", "NEURONS", "TOTAL_PARAMS", "CONFIG"])

for depth in range(args.mindepth, args.maxdepth+1, args.depthstep):
  for neurons in range(args.minneurons, args.maxneurons+1, args.neuronsstep):

    train = 'python training_scripts/train_taxi_parametric.py --depth={} --neurons={} --gpu=gpu0 --workers=1 --save-name=model{}-{} --iters={} --total-params-file=totalParams/totalParams[{}-{}]-[{}-{}].csv'.format(depth, neurons, depth, neurons, args.iters, args.mindepth, args.maxdepth, args.minneurons, args.maxneurons)
    process = subprocess.Popen(train.split())
    output, error = process.communicate()

    export = 'python savers/model_saver_taxi.py checkpoints/ppo/model{}-{}/checkpoint_{}/checkpoint-{} exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{}'.format(depth, neurons, '0'*(6-len(str(args.iters)))+str(args.iters), args.iters, depth, neurons, args.iters, depth, neurons)
    process = subprocess.Popen(export.split())
    output, error = process.communicate()

    convert = 'python converters/tflite_converter_taxi.py exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{} exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{}.tflite'.format(depth, neurons, args.iters, depth, neurons, depth, neurons, args.iters, depth, neurons)
    process = subprocess.Popen(convert.split())
    output, error = process.communicate()

    if not os.path.exists('datasets/dataset-model{}-{}.npy'.format(depth, neurons)):
      create_dataset = 'python dataset_creator.py {} datasets/dataset-model{}-{}'.format(args.dim, depth, neurons)
      process = subprocess.Popen(create_dataset.split())
      output, error = process.communicate()


    quant8b = 'python quantizers/quantizer8b_taxi.py datasets/dataset-model{}-{}.npy exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{} exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{}-quant8.tflite'.format(depth, neurons, depth, neurons, args.iters, depth, neurons, depth, neurons, args.iters, depth, neurons)
    process = subprocess.Popen(quant8b.split())
    output, error = process.communicate()


    edge_tpu = 'edgetpu_compiler exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{}-quant8.tflite'.format(depth, neurons, args.iters, depth, neurons)
    process = subprocess.Popen(edge_tpu.split())
    output, error = process.communicate()

    move_tpu_files = 'mv checkpoint-{}-model{}-{}-quant8_edgetpu.tflite exported_models/depth-neurons/model{}-{}/checkpoint-{}-model{}-{}-quant8_edgetpu.tflite'.format(args.iters, depth, neurons, depth, neurons, args.iters, depth, neurons)
    process = subprocess.Popen(move_tpu_files.split())
    output, error = process.communicate()

    clear_log = 'rm checkpoint-{}-model{}-{}-quant8_edgetpu.log'.format(args.iters, depth, neurons)
    process = subprocess.Popen(clear_log.split())
    output, error = process.communicate()
