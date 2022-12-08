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

def main(model_deq = None, model_quant = None, env_name="Pong-v0", episodes = None):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-mdeq', '--model_deq', help='File path of dequant .tflite file.')
  parser.add_argument(
      '-mquant', '--model_quant', help='File path of quant .tflite file.')
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
  if model_deq is None:
    model_deq = args.model_deq
  if model_quant is None:
    model_quant = args.model_quant

  ## Getting distrib
  trainer = PPOTrainer(env = env_name, config={"framework": "tf2", "num_workers": 0})
  policy = trainer.get_policy()
  dist_class = policy.dist_class
  #print(dist_class)

  interpreter_deq = tflite.Interpreter(model_path=model_deq)
  interpreter_deq.allocate_tensors()
  input_details_deq = interpreter_deq.get_input_details()
  output_details_deq = interpreter_deq.get_output_details()
  print('Input details deq:', input_details_deq)
  print('Output details deq:', output_details_deq)

  interpreter_quant = tflite.Interpreter(model_path=model_quant)
  interpreter_quant.allocate_tensors()
  input_details_quant = interpreter_quant.get_input_details()
  output_details_quant = interpreter_quant.get_output_details()
  print('Input details:', input_details_quant)
  print('Output details:', output_details_quant)

  # Get image dim
  dim = input_details_deq[0]['shape'][1]

  env = wrappers.wrap_deepmind(gym.make(env_name), dim = dim)
  #env = gym.make(env_name)

  env.seed(0)
  prep = get_preprocessor(env.observation_space)(env.observation_space)
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge cpu is slow because it includes',
        'loading the model into Edge cpu memory.')

  timing_results=[]

  done = False
  this_step = 0
  steps = 0
  this_episode = 0
  reward_avg_deq = 0
  reward_avg_quant = 0
  while keep_going(steps, num_steps, this_episode, num_episodes):

    env.seed(this_episode)
    image = env.reset()
    image = prep.transform(image)
    image = image[np.newaxis, ...]
    print(image[0][0][0])
    #input("continuar")
    if input_details_deq[0]['dtype'] == np.uint8:
      image=np.uint8(image)

    done = False
    steps_this_episode = 0

    #print(input_details[0]['dtype'])
    #if input_details[0]['dtype'] == np.float32:
    # image=np.float32(image)
    #if input_details[0]['dtype'] == np.uint8:
    # image=np.uint8(image)

    interpreter_deq.set_tensor(input_details_deq[0]['index'], image)
    interpreter_quant.set_tensor(input_details_quant[0]['index'], image)

    this_step = 0
    reward_episode_deq = 0
    reward_episode_quant = 0
    while not done and keep_going(steps, num_steps, this_episode, num_episodes):
      interpreter_deq.invoke()
      interpreter_quant.invoke()
      output_data_deq = interpreter_deq.get_tensor(output_details_deq[0]['index'])
      output_data_quant = interpreter_quant.get_tensor(output_details_quant[0]['index'])
      action_deq = np.argmax(output_data_deq[0])
      action_quant = np.argmax(output_data_quant[0])
      print("Action deq:", action_deq)
      print("Action quant:", action_quant)
      if action_deq != action_quant:
        # Oly copy de state if actions are diffent (its very expensive)
        old_state = env.unwrapped.clone_state(include_rng=True)
        _, reward_quant, _, _ = env.step(action_quant)
        env.unwrapped.restore_full_state(old_state)
        image, reward_deq, done, prob = env.step(action_deq)
        reward_episode_deq += reward_deq
        reward_episode_quant += reward_quant
        print("Rwd deq:", reward_deq)
        print("Rwd quant:", reward_quant)
      else:
        image, reward, done, prob = env.step(action_deq)
        reward_episode_deq += reward
        reward_episode_quant += reward
      #print("Step {} --- Applied action {}. Returned observation: {}. Returned reward: {}. Probability: {}".format( this_step, action, image, reward, prob["prob"] ))
      this_step = this_step+1

      image = prep.transform(image)
      image = image[np.newaxis, ...]

      #if input_details[0]['dtype'] == np.float32:
      # image=np.float32(image)
      #if input_details[0]['dtype'] == np.uint8:
      # image=np.uint8(image)


      interpreter_deq.set_tensor(input_details_deq[0]['index'], image)
      interpreter_quant.set_tensor(input_details_quant[0]['index'], image)
      steps += 1
      ######################
      steps_this_episode += 1
    this_episode += 1
    reward_avg_deq += reward_episode_deq
    reward_avg_quant += reward_episode_quant
    print("Reward this episode deq", reward_episode_deq)
    print("Reward this episode quant", reward_episode_quant)
  if num_episodes is not None and num_episodes > 0:
    reward_avg_deq /= num_episodes
    reward_avg_quant /= num_episodes
  print("Reward avg deq:", reward_avg_deq)
  print("Reward avg quant:", reward_avg_quant)
  return reward_avg_deq, reward_avg_quant

if __name__ == '__main__':
  main()
