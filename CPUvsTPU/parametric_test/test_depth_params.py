import argparse
import subprocess
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../rollouts/'))
import rollout_tflite
import rollout_edge_tpu
import csv

thread_list = [1]
num_tpus_list = [1]
batches = range(50, 300, 20)

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
        '-s', '--steps', required=True, type=int, help='Number of inference steps'
)
parser.add_argument(
        '-b', '--batch', required=True, type=int, help='Input batch size'
)


args, unknown = parser.parse_known_args()
batch = args.batch if args.batch != None else 100

testFolder = 'test_results/depth-params/depth[{}-{}]-params[{}-{}]/'.format(args.mindepthpower, args.maxdepthpower, args.minparamspower, args.maxparamspower)
if not os.path.exists(testFolder):
  os.mkdir(testFolder)


depthRange = [2** pow for pow in range(args.mindepthpower, args.maxdepthpower+1, args.depthsteppower)]
paramsRange = [2 ** pow for pow in range(args.minparamspower, args.maxparamspower+1, args.paramsteppower)]

for num_tpus in num_tpus_list:
  tpuResultFileName = 'test_results/depth-params/depth[{}-{}]-params[{}-{}]/{}tpus_results.csv'.format(args.mindepthpower, args.maxdepthpower, args.minparamspower, args.maxparamspower, num_tpus)
  with open(tpuResultFileName, "w") as csvout:
    writer = csv.writer(csvout)
    header = ["Depth"]
    for params in paramsRange:
      header.append("{} params".format(params))
    writer.writerow(header)
    for depth in depthRange:
      row = [depth]
      for params in paramsRange:
        model = '../exported_models/depth-params/model{}-{}/checkpoint-1-model{}-{}-quant8_edgetpu.tflite'.format(depth, params, depth, params)
        if not os.path.exists(model):
          continue
        time_step_mean = rollout_edge_tpu.main(batch = batch, num_tpus = num_tpus, model=model, env_name="Taxi-v3")
        row.append(time_step_mean)
      writer.writerow(row)


for threads in thread_list:
  threadResultFileName = 'test_results/depth-params/depth[{}-{}]-params[{}-{}]/{}threads_results.csv'.format(args.mindepthpower, args.maxdepthpower, args.minparamspower, args.maxparamspower, threads)
  with open(threadResultFileName, "w") as csvout:
    writer = csv.writer(csvout)
    header = ["Depth"]
    for params in paramsRange:
      header.append("{} params".format(params))
    writer.writerow(header)
    for depth in depthRange:
      row = [depth]
      for params in paramsRange:
        model = '../exported_models/depth-params/model{}-{}/checkpoint-1-model{}-{}.tflite'.format(depth, params, depth, params)
        if not os.path.exists(model):
          continue
        time_step_mean = rollout_tflite.main(batch = batch, threads = threads, model=model, env_name="Taxi-v3")
        row.append(time_step_mean)
      writer.writerow(row)

