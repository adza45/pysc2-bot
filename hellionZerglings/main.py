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
#os.environ["SC2PATH"] = '/home/adam/Games/StarCraftII'
os.environ["SC2PATH"] = '/home/adamselement/.wine/drive_c/Program Files (x86)/StarCraft II'

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_LOAD_SCREEN = actions.FUNCTIONS.Load_screen.id
_UNLOADALLAT_SCREEN = actions.FUNCTIONS.UnloadAllAt_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_HOLDPOSITION = actions.FUNCTIONS.HoldPosition_quick.id

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTIONS = 20

class terranAgent(base_agent.BaseAgent):
	def __init__(self):
		super(terranAgent, self).__init__()
		self.previous_location = (42, 32)

	def can_do(self, obs, action):
		return action in obs.observation.available_actions

	def do_actions(self, obs, choice):

		if choice == 0:
			hellion = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Hellion]
			if len(hellion) > 0:
				hellion = hellion[0]
				return actions.FUNCTIONS.select_point("select_all_type", (hellion.x, hellion.y))

		elif choice == 1:
			medivac = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Medivac]
			if len(medivac) > 0:
				medivac = medivac[0]
				return actions.FUNCTIONS.select_point("select_all_type", (medivac.x, medivac.y))

		elif choice == 2:
			if self.can_do(obs, _SELECT_ARMY):
				return actions.FUNCTIONS.select_army("select")

		elif choice == 3:
			hellion = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Hellion]
			if len(hellion) > 0:
				hellion = hellion[0]
				if self.can_do(obs, _LOAD_SCREEN):
					return actions.FunctionCall(_LOAD_SCREEN, [_NOT_QUEUED, (hellion.x, hellion.y)])

		elif choice == 4:
			medivac = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Medivac]
			if len(medivac) > 0:
				medivac = medivac[0]
				if self.can_do(obs, _UNLOADALLAT_SCREEN):
					return actions.FunctionCall(_UNLOADALLAT_SCREEN, [_NOT_QUEUED, (medivac.x, medivac.y)])

		elif choice >= 5 and choice < 17:
			index = choice - 5
			points = np.zeros(shape=(12))
			points[index] = 1
			points = np.reshape(points, (-1, 4))
			location = np.where(points == 1)
			location = (((location[0] * 21.33) + 10.6), ((location[1] * 21) + 10.5))

			if self.can_do(obs, _MOVE_SCREEN):
				self.previous_location = (location[1], location[0])
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, (location[1], location[0])])

		elif choice == 17:
			if self.can_do(obs, _ATTACK_SCREEN):
					return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.previous_location])

		elif choice == 18:
			if self.can_do(obs, _HOLDPOSITION):
				return actions.FunctionCall(_HOLDPOSITION, [_NOT_QUEUED])

		elif choice == 19:
			return actions.FUNCTIONS.no_op()

		return actions.FUNCTIONS.no_op()

	def step(self, obs):
		super(terranAgent, self).step(obs)

		choice = random.randrange(ACTIONS)
		return self.do_actions(obs, choice)

def main(unused_argv):
	agent = terranAgent()
	try:
		while True:
			with sc2_env.SC2Env(
				  map_name="DefeatZealots2",
				  players=[sc2_env.Agent(sc2_env.Race.terran)],
				  agent_interface_format=features.AgentInterfaceFormat(
					feature_dimensions=features.Dimensions(screen=84, minimap=64),
					use_feature_units=True),
				  step_mul=1,
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
