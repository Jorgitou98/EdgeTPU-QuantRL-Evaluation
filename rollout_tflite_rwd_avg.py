import argparse

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

from statistics import mean
import csv


def make_interpreter(model_file, threads):
  return tflite.Interpreter(model_path=model_file, num_threads=threads)

def keep_going(steps, num_steps, episodes, num_episodes):
  if num_episodes:
    return episodes < num_episodes
  if num_steps:
    return steps < num_steps
  return True

def invoke(interpreter):
  interpreter.invoke()

def main(batch = 1, threads = 1, model = None, env_name="Pong-v0", steps = None):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', help='File path of first .tflite file.')
  parser.add_argument(
      '-i', '--input', required=False, help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-s', '--steps', type=int, default=10000,
      help='Number of times to run inference (overwriten by --episodes')
  parser.add_argument(
      '-e', '--episodes', type=int, default=0,
      help='Number of complete episodes to run (overrides --steps)')
  parser.add_argument(
      '-o', '--output', default = None,
      help= 'CSV file to store timing results')
  parser.add_argument(
      '-b', '--batch', default = 1,
      help= 'Size of batch for parallel inference')
  args, unknown = parser.parse_known_args()

  num_steps = steps if steps is not None else args.steps
  num_episodes = int(args.episodes)
  batch_size = int(batch/threads) if batch != 1 else 1
  if model is None: model = args.model
  print("Batch size", batch_size)

  batches_sizes = [batch_size for i in range(threads)]
  for i in range(batch - batch_size * threads):
    batches_sizes[i] += 1

  ## Getting distrib
  trainer = PPOTrainer(env = env_name, config={"framework": "tf2", "num_workers": 0})
  policy = trainer.get_policy()
  dist_class = policy.dist_class
  #print(dist_class)

  # Create TFLite interpreter
  interpreters_list = []
  for num_interpreter in range(threads):
        interpreters_list.append(make_interpreter(model, 1))
        input_dim = interpreters_list[num_interpreter].get_input_details()[0]['shape']
        input_dim[0] = batches_sizes[num_interpreter]
        print('Input dim:', input_dim)
        interpreters_list[num_interpreter].resize_tensor_input(0, input_dim)
        interpreters_list[num_interpreter].allocate_tensors()

  input_details = interpreters_list[0].get_input_details()
  output_details = interpreters_list[0].get_output_details()
  print('Input details:', input_details)
  print('Output details:', output_details)

  # Get image dim
  print('Input dim:', input_details[0]['shape'])


  # Get image dim
  dim = input_details[0]['shape'][1]

  env = wrappers.wrap_deepmind(gym.make(env_name), dim = dim)

  env.seed(0)
  # Create env
  #env = gym.make(env_name)

  prep = get_preprocessor(env.observation_space)(env.observation_space)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge cpu is slow because it includes',
        'loading the model into Edge cpu memory.')

  timing_results=[]

  done = False
  this_step = 0
  steps = 0
  episodes = 0
  reward_avg = 0
  while keep_going(steps, num_steps, episodes, num_episodes):
    image = env.reset()
    image = prep.transform(image)
    done = False
    steps_this_episode = 0

    print(input_details[0]['dtype'])
    if input_details[0]['dtype'] == np.float32:
      image=np.float32(image)
    if input_details[0]['dtype'] == np.uint8:
      image=np.uint8(image)

    for num_interpreter in range(threads):
      batch = [image for i in range(batches_sizes[num_interpreter])]
      print(image.shape)
      interpreters_list[num_interpreter].set_tensor(input_details[0]['index'], batch)
      #input("Continue")

    this_step = 0
    reward_episode = 0
    while not done and keep_going(steps, num_steps, episodes, args.episodes):

      #env.render()

      #input("Press to continue...")

      # Threads for parallel invoke of interpreters
      threads_list = []
      for num_interpreter in range(threads):
        threads_list.append(threading.Thread(target=invoke, args= (interpreters_list[num_interpreter],)))
      #interpreter.invoke()
      for thread in threads_list:
        thread.start()
      for thread in threads_list:
        thread.join()
      # We store time in ms
      for num_interpreter in range(threads):
        output_data = interpreters_list[num_interpreter].get_tensor(output_details[0]['index'])
        print('Output dim {} :'.format(num_interpreter), output_data.shape)
        #print('output data {}:'.format(num_interpreter), output_data)

      #dist = policy.dist_class(output_data, policy.model)
      #action = int(dist.sample())
      action = np.argmax(output_data[0])

      # Step environment and get reward and done information
      image, reward, done, prob = env.step(action)
      reward_episode += reward

      #print("Step {} --- Applied action {}. Returned observation: {}. Returned reward: {}. Probability: {}".format( this_step, action, image, reward, prob["prob"] ))
      this_step = this_step+1

      image = prep.transform(image)


      if input_details[0]['dtype'] == np.float32:
        image=np.float32(image)
      if input_details[0]['dtype'] == np.uint8:
        image=np.uint8(image)


      for num_interpreter in range(threads):
        batch = [image for i in range(batches_sizes[num_interpreter])]
        interpreters_list[num_interpreter].set_tensor(input_details[0]['index'], batch)

      steps += 1
      ######################
      steps_this_episode += 1
    episodes += 1
    reward_avg += reward_episode
  reward_avg /= episodes
  print("Reward avg:", reward_avg)
  return reward_avg

if __name__ == '__main__':
  main()
