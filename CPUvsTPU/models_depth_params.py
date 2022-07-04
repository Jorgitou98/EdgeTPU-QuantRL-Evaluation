
import argparse
import subprocess
import os
import csv
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
        '-dsp', '--depthsteppower', required=True, type=int, help='Step for increment depth step power'
)
parser.add_argument(
        '-mindp', '--mindepthpower', required=True, type=int, help='Integer indicating the min depth power'
)
parser.add_argument(
        '-maxdp', '--maxdepthpower', required=True, type=int, help='Integer indicating the max depth power'
)
parser.add_argument(
        '-psp', '--paramsteppower', required=True, type=int, help='Step for incrementing parameters power'
)
parser.add_argument(
        '-minpp', '--minparamspower', required=True, type=int, help='Integer indicating the min number of parameters power'
)

parser.add_argument(
        '-maxpp', '--maxparamspower', required=True, type=int, help='Integer indicating the max number of parameters power'
)
parser.add_argument(
        '-i', '--iters', required = True, type=int, help= 'Number of training iters to run'
)
parser.add_argument(
        '-d', '--dim', required = True, type=int, help= 'Input dimension'
)
args = parser.parse_args()

totalParamsFile = f'totalParams/totalParamDepthParamsPow[{args.mindepthpower}-{args.maxdepthpower}]-[{args.minparamspower}-{args.maxparamspower}].csv'

with open(totalParamsFile, 'w', newline='') as outcsv:
  writer = csv.writer(outcsv)
  writer.writerow(["DEPTH", "PARAMS", "REAL_TOTAL_PARAMS"])

for depth in [2** pow for pow in range(args.mindepthpower, args.maxdepthpower+1, args.depthsteppower)]:
  for params in [2 ** pow for pow in range(args.minparamspower, args.maxparamspower+1, args.paramsteppower)]:

    train = f'python train_taxi_parametric.py --depth={depth} --params={params} --gpu=gpu0 --workers=1 --save-name=model{depth}-{params} --iters={args.iters} --total-params-file={totalParamsFile}'
    process = subprocess.Popen(train.split())
    output, error = process.communicate()

    export = 'python model_saver_taxi.py checkpoints/ppo/model{}-{}/checkpoint_{}/checkpoint-{} exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{}'.format(depth, params, '0'*(6-len(str(args.iters)))+str(args.iters), args.iters, depth, params, args.iters, depth, params)
    process = subprocess.Popen(export.split())
    output, error = process.communicate()

    convert = 'python tflite_converter_taxi.py exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{} exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{}.tflite'.format(depth, params, args.iters, depth, params, depth, params, args.iters, depth, params)
    process = subprocess.Popen(convert.split())
    output, error = process.communicate()

    if not os.path.exists('datasets/dataset-model{}-{}.npy'.format(depth, params)):
      create_dataset = 'python dataset_creator.py {} datasets/dataset-model{}-{}'.format(args.dim, depth, params)
      process = subprocess.Popen(create_dataset.split())
      output, error = process.communicate()


    quant8b = 'python quantizer8b_taxi.py datasets/dataset-model{}-{}.npy exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{} exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{}-quant8.tflite'.format(depth, params, depth, params, args.iters, depth, params, depth, params, args.iters, depth, params)
    process = subprocess.Popen(quant8b.split())
    output, error = process.communicate()


    edge_tpu = 'edgetpu_compiler exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{}-quant8.tflite'.format(depth, params, args.iters, depth, params)
    process = subprocess.Popen(edge_tpu.split())
    output, error = process.communicate()

    move_tpu_files = 'mv checkpoint-{}-model{}-{}-quant8_edgetpu.tflite exported_models/depth-params/model{}-{}/checkpoint-{}-model{}-{}-quant8_edgetpu.tflite'.format(args.iters, depth, params, depth, params, args.iters, depth, params)
    process = subprocess.Popen(move_tpu_files.split())
    output, error = process.communicate()

    clear_log = 'rm *.log'
    process = subprocess.Popen(move_tpu_files.split())
    output, error = process.communicate()
