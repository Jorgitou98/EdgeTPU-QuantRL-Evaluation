import argparse
import subprocess
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
import rollout_tflite
import rollout_edge_tpu
import csv

thread_list = [1]
num_tpus_list = [1]
batches = range(50, 300, 20)

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
        '-s', '--steps', required=True, type=int, help='Number of inference steps'
)
parser.add_argument(
        '-b', '--batch', required=True, type=int, help='Input batch size'
)


args, unknown = parser.parse_known_args()
batch = args.batch if args.batch != None else 100

testFolder = 'test_results/depth-neurons/depth[{}-{}]-neurons[{}-{}]/'.format(args.mindepth, args.maxdepth, args.minneurons, args.maxneurons)
if not os.path.exists(testFolder):
  os.mkdir(testFolder)

for num_tpus in num_tpus_list:
  tpuResultFileName = 'test_results/depth-neurons/depth[{}-{}]-neurons[{}-{}]/{}tpus_results.csv'.format(args.mindepth, args.maxdepth, args.minneurons, args.maxneurons, num_tpus)
  with open(tpuResultFileName, "w") as csvout:
    writer = csv.writer(csvout)
    header = ["Depth"]
    for neurons in range(args.minneurons, args.maxneurons+1, args.neuronsstep):
      header.append("{} neurons".format(neurons))
    writer.writerow(header)
    for depth in range(args.mindepth, args.maxdepth+1, args.depthstep):
      row = [depth]
      for neurons in range(args.minneurons, args.maxneurons+1, args.neuronsstep):
        model = './exported_models/depth-neurons/model{}-{}/checkpoint-1-model{}-{}-quant8_edgetpu.tflite'.format(depth, neurons, depth, neurons)
        if not os.path.exists(model):
          continue
        time_step_mean = rollout_edge_tpu.main(batch = batch, num_tpus = num_tpus, model=model, env_name="Taxi-v3")
        row.append(time_step_mean)
      writer.writerow(row)


for threads in thread_list:
  threadResultFileName = 'testResults/depth-neurons/depth[{}-{}]-neurons[{}-{}]/{}threads_results.csv'.format(args.mindepth, args.maxdepth, args.minneurons, args.maxneurons, threads)
  with open(threadResultFileName, "w") as csvout:
    writer = csv.writer(csvout)
    header = ["Depth"]
    for neurons in range(args.minneurons, args.maxneurons+1, args.neuronsstep):
      header.append("{} neurons".format(neurons))
    writer.writerow(header)
    for depth in range(args.mindepth, args.maxdepth+1, args.depthstep):
      row = [depth]
      for neurons in range(args.minneurons, args.maxneurons+1, args.neuronsstep):
        model = './exported_models/depth-neurons/model{}-{}/checkpoint-1-model{}-{}.tflite'.format(depth, neurons, depth, neurons)
        if not os.path.exists(model):
          continue
        time_step_mean = rollout_tflite.main(batch = batch, threads = threads, model=model, env_name="Taxi-v3")
        row.append(time_step_mean)
      writer.writerow(row)

