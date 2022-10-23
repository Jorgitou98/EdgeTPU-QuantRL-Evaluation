import argparse

from PIL import Image

import tflite_runtime.interpreter as tflite
import numpy as np
import platform
import tensorflow as tf
import ray.rllib.env.wrappers.atari_wrappers as wrappers
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.agents.ppo import PPOTrainer
import gym

def keep_going(steps, num_steps, episodes, num_episodes):
  if num_episodes:
    return episodes < num_episodes
  if num_steps:
    return steps < num_steps
  return True

def main(model = None, env_name="Pong-v0", episodes = None):
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
  args, unknown = parser.parse_known_args()

  num_steps = int(args.steps)
  num_episodes = episodes if episodes is not None else int(args.episodes)
  print(num_episodes)
  if model is None:
    model = args.model

  ## Getting distrib
  trainer = PPOTrainer(env = env_name, config={"framework": "tf2", "num_workers": 0})
  policy = trainer.get_policy()
  dist_class = policy.dist_class
  #print(dist_class)

  interpreter = tflite.Interpreter(model_path=model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
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
  this_episode = 0
  reward_avg = 0
  while keep_going(steps, num_steps, this_episode, num_episodes):

    env.seed(this_episode)
    image = env.reset()
    image = prep.transform(image)
    image = image[np.newaxis, ...]

    done = False
    steps_this_episode = 0

    print(input_details[0]['dtype'])
    #if input_details[0]['dtype'] == np.float32:
    # image=np.float32(image)
    #if input_details[0]['dtype'] == np.uint8:
    # image=np.uint8(image)

    interpreter.set_tensor(input_details[0]['index'], image)

    this_step = 0
    reward_episode = 0
    while not done and keep_going(steps, num_steps, this_episode, num_episodes):
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      print("Output:", output_data)
      action = np.argmax(output_data[0])
      print("Action:", action)

      # Step environment and get reward and done information
      image, reward, done, prob = env.step(action)
      reward_episode += reward

      #print("Step {} --- Applied action {}. Returned observation: {}. Returned reward: {}. Probability: {}".format( this_step, action, image, reward, prob["prob"] ))
      this_step = this_step+1

      image = prep.transform(image)
      image = image[np.newaxis, ...]

      #if input_details[0]['dtype'] == np.float32:
      # image=np.float32(image)
      #if input_details[0]['dtype'] == np.uint8:
      # image=np.uint8(image)


      interpreter.set_tensor(input_details[0]['index'], image)
      steps += 1
      ######################
      steps_this_episode += 1
    this_episode += 1
    reward_avg += reward_episode
    print("Reward this episode", reward_episode)
  reward_avg /= episodes
  print("Reward avg:", reward_avg)
  return reward_avg

if __name__ == '__main__':
  main()
