import argparse

from PIL import Image
from numpy import linalg as LA
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

def main(model_quant = None, model_dequant = None, env_name="Pong-v0", steps = None):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-mdq', '--model-dequant', help='File path of no quant .tflite file no')
  parser.add_argument(
      '-mq', '--model-quant', help='File path of quant .tflite file')
  parser.add_argument(
      '-i', '--input', required=False, help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-s', '--steps', type=int, default=10000,
      help='Number of times to run inference (overwriten by --episodes')
  parser.add_argument(
      '-o', '--output', default = None,
      help= 'CSV file to store timing results')
  args, unknown = parser.parse_known_args()

  num_steps = steps if steps is not None else int(args.steps)

  if model_quant is None:
    model_quant = args.model_quant

  if model_dequant is None:
    model_dequant = args.model_dequant


  ## Getting distrib
  #trainer = PPOTrainer(env = env_name, config={"framework": "tf2", "num_workers": 0})
  #trainer = PPOTrainer(env = env_name)
  #policy = trainer.get_policy()
  #dist_class = policy.dist_class
  #print(dist_class)

  interpreter_dequant = tflite.Interpreter(model_path=model_dequant)
  interpreter_dequant.allocate_tensors()
  input_details_dequant = interpreter_dequant.get_input_details()
  output_details_dequant = interpreter_dequant.get_output_details()
  print('Input details:', input_details_dequant)
  print('Output details:', output_details_dequant)

  interpreter_quant = tflite.Interpreter(model_path=model_quant)
  interpreter_quant.allocate_tensors()
  input_details_quant = interpreter_quant.get_input_details()
  output_details_quant = interpreter_quant.get_output_details()

  # Get image dim
  print('Input dim:', input_details_dequant[0]['shape'])


  # Get image dim
  dim = input_details_dequant[0]['shape'][1]

  env = wrappers.wrap_deepmind(gym.make(env_name), dim = dim)

  # Create env
  #env = gym.make(env_name)

  prep = get_preprocessor(env.observation_space)(env.observation_space)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge cpu is slow because it includes',
        'loading the model into Edge cpu memory.')
  rel_norm_acumm = 0
  
  seed = 0
  done = True
  for step in range(num_steps):
    if done:
      env.seed(seed)
      image = env.reset()
      seed+=1
    image = prep.transform(image)
    image = image[np.newaxis, ...]

    interpreter_dequant.set_tensor(input_details_dequant[0]['index'], image)
    interpreter_quant.set_tensor(input_details_quant[0]['index'], image)

    interpreter_dequant.invoke()
    interpreter_quant.invoke()
    output_data_dequant = interpreter_dequant.get_tensor(output_details_dequant[0]['index'])
    output_data_quant = interpreter_quant.get_tensor(output_details_quant[0]['index'])
    print("_"*50)
    print(output_data_dequant)
    print(output_data_quant)
    print(output_data_dequant - output_data_quant)
    print(LA.norm(output_data_dequant - output_data_quant)/LA.norm(output_data_dequant))
    rel_norm_acumm += LA.norm(output_data_dequant - output_data_quant)/LA.norm(output_data_dequant)
    action = np.argmax(output_data_dequant[0])
    image, _, done, _ = env.step(action)
  return rel_norm_acumm/num_steps

if __name__ == '__main__':
  main()
