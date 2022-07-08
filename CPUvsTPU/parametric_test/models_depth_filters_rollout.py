import argparse
import subprocess
import os
import csv
import sys
sys.path.append(os.path.join(sys.path[0], '../rollouts'))
import rollout_tflite
import rollout_edge_tpu


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
parser.add_argument(
        '-s', '--steps', type=int, default=10000, help='Number of times to run inference'
)

args = parser.parse_args()


for depth in range(args.mindepth, args.maxdepth+1, args.depthstep):
  resultTPU = [depth]
  resultCPU = [depth]
  for filters in range(args.minfilters, args.maxfilters+1, args.filtersstep):

    train = f'python ../training_scripts/train_pong_parametric.py --depth={depth} --filters={filters} --gpu=gpu0 --workers=1 --save-name=model_depth_filters{depth}-{filters} --iters={args.iters}'
    process = subprocess.Popen(train.split())
    process.communicate()

    checkpoint_number = '0'*(6-len(str(args.iters)))+str(args.iters)
    export = f'python ../savers/model_saver_pong.py ../checkpoints/ppo/model_depth_filters{depth}-{filters}/checkpoint_{checkpoint_number}/checkpoint-{args.iters} ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}'
    process = subprocess.Popen(export.split())
    process.communicate()

    convert = f'python ../converters/tflite_converter_pong.py ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters} ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}.tflite'
    process = subprocess.Popen(convert.split())
    process.communicate()

    if not os.path.exists(f'datasets/dataset-model_pong_{args.dim}.npy'):
      create_dataset = f'python datasets/dataset_creator.py {args.dim} datasets/dataset-model_pong_{args.dim} Pong-v0'
      process = subprocess.Popen(create_dataset.split())
      process.communicate()


    quant8b = f'python ../quantizers/quantizer8b_pong.py datasets/dataset-model_pong_{args.dim}.npy ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters} ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}-quant8.tflite'
    process = subprocess.Popen(quant8b.split())
    process.communicate()

    edge_tpu = f'edgetpu_compiler ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}-quant8.tflite'
    process = subprocess.Popen(edge_tpu.split())
    process.communicate()

    move_tpu_files = f'mv checkpoint-{args.iters}-model{depth}-{filters}-quant8_edgetpu.tflite ../exported_models/depth-filters/model{depth}-{filters}/checkpoint-{args.iters}-model{depth}-{filters}-quant8_edgetpu.tflite'
    process = subprocess.Popen(move_tpu_files.split())
    process.communicate()

    rm_log = f'rm checkpoint-{args.iters}-model{depth}-{filters}-quant8_edgetpu.log'
    process = subprocess.Popen(rm_log.split())
    process.communicate()

    # Inferences in CPU and TPU

    modelTPU = f'../exported_models/depth-filters/model{depth}-{filters}/checkpoint-1-model{depth}-{filters}-quant8_edgetpu.tflite'
    timeTPU = rollout_edge_tpu.main(batch = 1, num_tpus = 1, model=modelTPU, env_name="Pong-v0", steps = args.steps)
    resultTPU.append(timeTPU)

    modelCPU = f'../exported_models/depth-filters/model{depth}-{filters}/checkpoint-1-model{depth}-{filters}.tflite'
    timeCPU = rollout_tflite.main(batch = 1, num_threads = 1, model=modelCPU, env_name="Pong-v0", steps = args.steps)
    resultCPU.append(timeCPU)


    # Remove generated files

    rm_training_results = f'rm ../training_results/ppo/model_depth_filters{depth}-{filters}.csv ../training_results/ppo/model_depth_filters{depth}-{filters}.json'
    process = subprocess.Popen(rm_training_results.split())
    process.communicate()

    rm_checkpoints = f'rm -r ../checkpoints/ppo/model_depth_filters{depth}-{filters}/'
    process = subprocess.Popen(rm_checkpoints.split())
    process.communicate()

    rm_models = f'rm -r ../exported_models/depth-filters/model{depth}-{filters}/'
    process = subprocess.Popen(rm_models.split())
    process.communicate()

    rm_ray_results = f'rm -r ray_results/model_depth_filters{depth}-{filters}/'
    process = subprocess.Popen(rm_ray_results.split())
    process.communicate()

  writerTPU.writerow(resultTPU)
  writerCPU.writerow(resultCPU)

timeTPUout.close()
timeCPUout.close()
