from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np
import pandas as pd
import random
import cv2
import os
import math
import time
import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import pickle
from collections import deque

#os.environ["SC2PATH"] = '/media/adamselement/SAMSUNG SSD/Projects/match-runner/sc2-bot-match-runner/StarCraftII'
os.environ["SC2PATH"] = '/home/adam/Games/StarCraftII'

class terranAgent(base_agent.BaseAgent):
	def __init__(self):
		super(terranAgent, self).__init__()

	def step(self):
		super(terranAgent, self).step(obs)

		return actions.FUNCTIONS.no_op()

def main(unused_argv):
	agent = terranAgent()
	try:
		while True:
			with sc2_env.SC2Env(
				  map_name="test",
				  players=[sc2_env.Agent(sc2_env.Race.terran)],
				  agent_interface_format=features.AgentInterfaceFormat(
					feature_dimensions=features.Dimensions(screen=84, minimap=64),
					use_feature_units=True),
				  step_mul=16,
				  game_steps_per_episode=0,
				  visualize=True) as env:

				agent.setup(env.observation_spec(), env.action_spec())

				timesteps = env.reset()
				agent.reset()

				while True:
				  step_actions = [agent.step(timesteps[0])]
				  timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
  app.run(main)
