import argparse
import subprocess
import os
import rollout_tflite
import rollout_edge_tpu
import rollout_edge_tpu_pipeline
import csv

thread_list = [1]
batches = range(1, 201, 50)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        '-m', '--model', required=True, type=int, choices=[i for i in range(1,9)], help='Integer with the number of model'
)
parser.add_argument(
        '-s', '--steps', required=False, type=int, help='Number of inference steps per batch mean time'
)
parser.add_argument(
        '-nt', '--num-tpus-list', required=False, nargs="+", default="1", type=int, help='Number of TPUs used'
)
parser.add_argument(
        '-nth', '--num-thread-list', required=False, nargs="+", default="1", type=int, help='Number of CPU threads permitted'
)


args, unknown = parser.parse_known_args()

base_path = f"exported_models/batch_test/model{args.model}"
tpu_path = base_path + "-quant8_edgetpu.tflite"
tpu_pipeline_path = base_path + "-quant8"
cpu_path = base_path + ".tflite"

for num_tpus in args.num_tpus_list:
  # No tiene sentido el pipeline de una sola tpu
  if num_tpus == 1:
    continue
  with open(f'results/batch_test/model{args.model}_{num_tpus}tpus_pipeline.csv', "w") as csvout:
    writer = csv.writer(csvout)
    writer.writerow(["Batch", "Mean time {} TPUs pipeline".format(num_tpus)])
    for batch in batches:
      time_step_mean = rollout_edge_tpu_pipeline.main(batch = batch, num_tpus = num_tpus, model = tpu_pipeline_path, steps=args.steps)
      writer.writerow([batch, time_step_mean])

for num_tpus in args.num_tpus_list:
  with open(f'results/batch_test/model{args.model}_{num_tpus}tpus.csv', "w") as csvout:
    writer = csv.writer(csvout)
    writer.writerow(["Batch", "Mean time {} TPUs".format(num_tpus)])
    for batch in batches:
      time_step_mean = rollout_edge_tpu.main(batch = batch, num_tpus = num_tpus, model = tpu_path, steps=args.steps)
      writer.writerow([batch, time_step_mean])

for threads in args.num_thread_list:
  with open(f'results/batch_test/model{args.model}_cpu_{threads}threads.csv', "w") as csvout:
    writer = csv.writer(csvout)
    writer.writerow(["Batch", "Mean time CPU {} threads".format(threads)])
    for batch in batches:
      time_step_mean = rollout_tflite.main(batch = batch, threads = threads, model = cpu_path, steps=args.steps)
      writer.writerow([batch, time_step_mean])
