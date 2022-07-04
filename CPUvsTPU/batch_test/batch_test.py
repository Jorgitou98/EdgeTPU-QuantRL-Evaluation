import argparse
import subprocess
import os
import rollout_tflite
import rollout_edge_tpu
import csv

thread_list = [1, 2, 4, 8, 16]
num_tpus_list = [1, 2, 4, 8]
batches = range(50, 300, 20)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        '-nm', '--nummodels', required=True, type=int, choices=[i for i in range(1,9)], help='Integer with the number of model'
)
args, unknown = parser.parse_known_args()


for num_tpus in num_tpus_list:
  with open('batch_test_results/model{}_{}tpus_results.csv'.format(args.nummodels, num_tpus), "w") as csvout:
    writer = csv.writer(csvout)
    writer.writerow(["Batch", "Mean time {} tpus 8b".format(num_tpus)])
    for batch in batches:
      time_step_mean = rollout_edge_tpu.main(batch = batch, num_tpus = num_tpus)
      writer.writerow([batch, time_step_mean])

for threads in thread_list:
  with open('batch_test_results/model{}_cpu_{}thread_results.csv'.format(args.nummodels, threads), "w") as csvout:
    writer = csv.writer(csvout)
    writer.writerow(["Batch", "Mean time cpu 32b {} threads".format(threads)])
    for batch in batches:
      time_step_mean = rollout_tflite.main(batch = batch, threads = threads)
      writer.writerow([batch, time_step_mean])

