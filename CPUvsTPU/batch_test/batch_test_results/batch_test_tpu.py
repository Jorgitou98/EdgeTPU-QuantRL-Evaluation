import argparse
import subprocess
import os
import rollout_edge_tpu
import csv
import sys

num_tpus_list = [1, 2, 4, 8]
batches = range(50, 300, 20)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        '-nm', '--nummodel', required=True, type=int, choices=[i for i in range(1,9)], help='Integer with the number of model'
)
args, unknown = parser.parse_known_args()


for num_tpus in num_tpus_list:
  with open('batchTestResults/model{}_{}tpus_results.csv'.format(args.nummodel, num_tpus), "w") as csvout:
    writer = csv.writer(csvout)
    writer.writerow(["Batch", "Mean time {} tpus 8b".format(num_tpus)])
    for batch in batches:
      time_step_mean = rollout_edge_tpu.main(batch = batch, num_tpus = num_tpus)
      writer.writerow([batch, time_step_mean])
