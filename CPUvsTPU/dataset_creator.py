import ray
import ray.rllib.agents.ppo as ppo
import numpy as np
import sys

import ray.rllib.env.wrappers.atari_wrappers as wrappers
import gym

dim = int(sys.argv[1])
dataset_name = sys.argv[2]

env = wrappers.wrap_deepmind(gym.make('Pong-v0'), dim = dim)
obs = env.reset()
print("Observations shape: ", obs.shape)
with open(dataset_name + '.npy', 'wb') as f:
    for _ in range(500):
        np.save(f, obs)
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
