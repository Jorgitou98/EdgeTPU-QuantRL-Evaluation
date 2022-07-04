import argparse
import time

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

from statistics import mean
import csv


def make_interpreter(model_file):
  return tflite.Interpreter(model_path=model_file)

def keep_going(steps, num_steps, episodes, num_episodes):
  if num_episodes:
    return episodes < num_episodes
  if num_steps:
    return steps < num_steps
  return True

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of first .tflite file.')
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
  args = parser.parse_args()

  num_steps = int(args.steps)
  num_episodes = int(args.episodes)

  # Getting distrib
  trainer = PPOTrainer(env="Taxi-v3", config={"framework": "tf2", "num_workers": 0})
  policy = trainer.get_policy()
  dist_class = policy.dist_class
  print(dist_class)

  # Create TFLite interpreter
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  print('Input details: ', input_details)
  print('Output details: ', output_details)

  input("Continue Enter...")

  # Create env
  env = gym.make('Taxi-v3')

  prep = get_preprocessor(env.observation_space)(env.observation_space)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')

  timing_results=[]

  done = False
  this_step = 0
  steps = 0
  episodes = 0
  reward_avg = 0
  rewards = []
  while keep_going(steps, num_steps, episodes, num_episodes):
    env.seed(episodes)
    image = env.reset()

    image = prep.transform(image)

    done = False
    steps_this_episode = 0
    image = image[np.newaxis, ...]
    if input_details[0]['dtype'] == np.uint8:
      image=np.uint8(image)

    interpreter.set_tensor(input_details[0]['index'], image)

    this_step = 0
    reward_episode = 0
    while not done and keep_going(steps, args.steps, episodes, args.episodes):
      env.render()

      #input("Press to continue...)
      interpreter.invoke()

      output_data = interpreter.get_tensor(output_details[0]['index'])
      print(output_data)

      #dist = policy.dist_class(output_data, policy.model)
      #action = int(dist.sample())
      action = np.argmax(output_data)
      # Step environment and get reward and done information
      image, reward, done, prob = env.step(action)
      reward_episode += reward
      #print("Step {} --- Applied action {}. Returned observation: {}. Returned reward: {}. Probability: {}".format( this_step, action, image, reward, prob["prob"] ))
      this_step = this_step+1

      image = prep.transform(image)

      # Place new image as the new model's input
      image = image[np.newaxis, ...]

      if input_details[0]['dtype'] == np.uint8:
        image=np.uint8(image)

      interpreter.set_tensor(input_details[0]['index'], image)

      steps += 1
      ######################
      steps_this_episode += 1
    if(reward_episode > -200):
      rewards.append((episodes, reward_episode))
    episodes += 1
    reward_avg += reward_episode

  reward_avg /= episodes
  print("Rewards over -200:", rewards)
  print("Reward avg:", reward_avg)

if __name__ == '__main__':
  main()
