import argparse
import time
import pycoral.pipeline.pipelined_model_runner as pipeline
from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np
import platform
import tensorflow as tf
from scipy.special import softmax
import ray.rllib.env.wrappers.atari_wrappers as wrappers
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.agents.ppo import PPOTrainer
import gym
import threading
from pycoral.adapters import common
from statistics import mean
import csv
import os

def make_runner(model_file, num_interpreters):
  interpreters = [tflite.Interpreter(model_path=model_file + f"_segment_{num_dev}_of_{num_interpreters}_edgetpu.tflite",
                  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', options={"device": f":{num_dev}"})]) for num_dev in range(num_interpreters)]
  for interpreter in interpreters:
    interpreter.allocate_tensors()
  return pipeline.PipelinedModelRunner(interpreters)

def make_inference(model, env_name, batch, num_tpus):
  ## Getting distrib
  trainer = PPOTrainer(env = env_name, config={"framework": "tf2", "num_workers": 0})
  policy = trainer.get_policy()
  dist_class = policy.dist_class

  # Create TFLite interpreters
  runner = make_runner(model, num_tpus)

  size = common.input_size(runner.interpreters()[0])
  name = common.input_details(runner.interpreters()[0], 'name')
  print("Input size:", size)
  print("Input details:", name)

  env = wrappers.wrap_deepmind(gym.make(env_name), dim = 84)

  prep = get_preprocessor(env.observation_space)(env.observation_space)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')

  image = env.reset()
  image = prep.transform(image)

  def producer():
    for _ in range(batch):
      runner.push({name: image})
    runner.push({})

  def consumer():
    output_details = runner.interpreters()[-1].get_output_details()[0]
    print(output_details)
    while True:
      result = runner.pop()
      if not result:
        break
      values, = result.values()
      print(values)

  start = time.perf_counter()
  producer_thread = threading.Thread(target=producer)
  consumer_thread = threading.Thread(target=consumer)
  producer_thread.start()
  consumer_thread.start()
  producer_thread.join()
  consumer_thread.join()
  step_time_ms = (time.perf_counter() - start) * 1000
  print("Step time in ms:", step_time_ms)
  return step_time_ms

def main(batch = 1, num_tpus = 1, model = None, env_name="Pong-v0", steps = 1):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', help='File path of first .tflite file.')
  parser.add_argument(
      '-s', '--steps', type=int, default=10000,
      help='Number of times to run inference (overwriten by --episodes')
  parser.add_argument(
      '-b', '--batch', default = 1,
      help= 'Size of batch for parallel inference')
  args, unknown = parser.parse_known_args()

  #fd = os.open('/dev/null',os.O_WRONLY)
  #os.dup2(fd,2)

  if model is None:
    model = args.model
  print("Batch size", batch)

  total_time_ms = 0
  for step in range(steps):
    step_time_ms = make_inference(model = model, env_name = env_name, batch = batch, num_tpus = num_tpus)
    total_time_ms += step_time_ms
  avg_time_ms = total_time_ms/steps

  print('Average inference time (%d steps with %d batch size): %.5fms' %
        (steps, batch, avg_time_ms))
  input("Continuar...")
  return avg_time_ms

if __name__ == '__main__':
  main()
