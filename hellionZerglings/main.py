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
#os.environ["SC2PATH"] = '/home/adamselement/.wine/drive_c/Program Files (x86)/StarCraft II'

#game parameters
ACTIONS = 20
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 8000000
FINAL_EPSILON = 0.01
INITIAL_EPSILON = 0.99
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
GAME = 'sc2'
LEARNING_RATE = 0.00001
img_rows , img_cols = 64,84
img_channels = 4

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_LOAD_SCREEN = actions.FUNCTIONS.Load_screen.id
_UNLOADALLAT_SCREEN = actions.FUNCTIONS.UnloadAllAt_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_HOLDPOSITION = actions.FUNCTIONS.HoldPosition_quick.id

_NOT_QUEUED = [0]
_QUEUED = [1]

class terranAgent(base_agent.BaseAgent):
	def __init__(self):
		super(terranAgent, self).__init__()
		self.previous_location = (42, 32) #initial attack location, for attack action 17
		self.score = 0 #Initial Score
		self.previous_score = 0 #Initialize previous score
		self.previous_zerglings = []
		self.total_reward = 0
		self.current_reward = 0
		self.top_random_reward = 0
		self.top_explore_reward = 0

	#Based From
	#https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/LICENSE
	#https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	def createNetwork(self):
		# network weights
		W_conv1 = self.weight_variable([8, 8, 4, 32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4, 4, 32, 64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3, 3, 64, 64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([1536, 512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512, ACTIONS])
		b_fc2 = self.bias_variable([ACTIONS])

		# input layer
		s = tf.placeholder("float", [None, 64, 84, 4])

		# hidden layers
		h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
		#h_pool2 = max_pool_2x2(h_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
		#h_pool3 = max_pool_2x2(h_conv3)

		h_conv3_flat = tf.reshape(h_conv3, [-1, 1536])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

		# readout layer
		readout = tf.matmul(h_fc1, W_fc2) + b_fc2

		return s, readout, h_fc1

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

	def grab_screen(self, obs):
		game_data = np.zeros((64, 84, 3), np.uint8)

		for unit in obs.observation.feature_units:
			#print("UNIT: {}".format(dir(unit)))
			#color = (math.ceil(unit.unit_type / 2), math.ceil(unit.unit_type / 2), math.ceil(unit.unit_type / 2))
			if unit.unit_type == units.Terran.Medivac:
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius * 3), (255,255,255))
				if unit.is_selected:
					cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius * 3) + 2, (160,160,160))
			elif unit.unit_type == units.Terran.Hellion:
				color = 191 + ((unit.health / 90) * 64)
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius * 2), (color,color,color))
				if unit.is_selected:
					cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius * 2) + 2, (160,160,160))
			elif unit.unit_type == units.Zerg.Zergling:
				color = 64 + ((unit.health / 35) * 64)
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius * 2), (color,color,color))

		image = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)

		cv2.waitKey(1)

		return image

	def step(self, obs):
		super(terranAgent, self).step(obs)

		if obs.last():
			self.current_reward = 0
			self.terminal = True


		if obs.first():
			self.current_reward = 0
			#self.previous_zerglings = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Zerg.Zergling]

		reward = 0

		# zerglings = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Zerg.Zergling]
		# for x in range(0, len(zerglings)):
		# 	if len(zerglings) == len(self.previous_zerglings):
		# 		if zerglings[x][2] < self.previous_zerglings[x][2]:
		# 			reward = reward + .001#0.005
		#
		# self.previous_zerglings = zerglings

		self.score = obs.observation['score_cumulative'][0]
		if self.score > self.previous_score:
			#self.r_t = BEACON_REWARD
			self.r_t = self.score - self.previous_score
		else:
			#self.r_t = -0.1
			if reward == 0:
				reward = -0.1

			self.r_t = reward

		#print("r_t: {}, score: {}, previous score: {}, total reward: {}".format(self.r_t, self.score, self.previous_score, self.total_reward))
		self.previous_score = self.score

		self.x_t1 = self.grab_screen(obs)
		self.x_t1 = np.reshape(self.x_t1, (64, 84, 1))
		self.s_t1 = np.append(self.x_t1, self.s_t[:, :, :3], axis=2)
		self.last_time = time.time()

		resized = cv2.resize(self.s_t1, dsize=None, fx=2, fy=2)
		cv2.imshow("Stacked Image", resized)

		# store the transition in D
		self.D.append((self.s_t, self.a_t, self.r_t, self.s_t1, self.terminal))

		if len(self.D) > REPLAY_MEMORY:
			self.D.popleft()

		if self.t > OBSERVE:
			minibatch = random.sample(self.D, BATCH)

			# get the batch variables
			s_j_batch = [d[0] for d in minibatch]
			a_batch = [d[1] for d in minibatch]
			r_batch = [d[2] for d in minibatch]
			s_j1_batch = [d[3] for d in minibatch]

			y_batch = []
			readout_j1_batch = self.readout.eval(feed_dict = {self.s : s_j1_batch})
			for i in range(0, len(minibatch)):
				self.terminal = minibatch[i][4]
				# if terminal, only equals reward
				if self.terminal:
					y_batch.append(r_batch[i])
				else:
					y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

			# perform gradient step
			self.train_step.run(feed_dict = {
				self.y : y_batch,
				self.a : a_batch,
				self.s : s_j_batch}
			)

		self.s_t = self.s_t1
		self.t = self.t + 1
		self.total_reward += self.r_t

		if self.t % 10000 == 0:
			self.saver.save(self.sess, 'saved_networks/' + GAME + '-dqn', global_step = self.t)

		self.current_reward = self.current_reward +self.r_t

		state = ""
		if self.t <= OBSERVE:
			state = "observe"
			if self.current_reward > self.top_random_reward:
				self.top_random_reward = self.current_reward
		elif self.t > OBSERVE and self.t <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		if state == "explore" or state == "train":
			if self.current_reward > self.top_explore_reward:
				self.top_explore_reward = self.current_reward

		print("TIMESTEP", self.t, "TOTAL REWARD", self.total_reward, "/ STATE", state, \
			"/ EPSILON", self.epsilon, "/ ACTION", self.action_index, "/ REWARD", self.r_t, \
			"/ Q_MAX %e" % np.max(self.readout_t), "/ TOP RANDOM", self.top_random_reward, "TOP EXP/TRN", self.top_explore_reward)

		self.terminal = False

		# choose an action epsilon greedily
		self.readout_t = self.readout.eval(feed_dict={self.s : [self.s_t]})[0]
		self.a_t = np.zeros([ACTIONS])
		self.action_index = 0
		if self.t % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				#print("----------Random Action----------")
				self.action_index = random.randrange(ACTIONS)
				self.a_t[random.randrange(ACTIONS)] = 1
			else:
				self.action_index = np.argmax(self.readout_t)
				self.a_t[self.action_index] = 1
		else:
			self.a_t[0] = 1 # do nothing

		# scale down epsilon
		if self.epsilon > FINAL_EPSILON and self.t > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		return self.do_actions(obs, np.where(self.a_t == 1)[0][0])

def main(unused_argv):
	agent = terranAgent()
	try:
		while True:
			with sc2_env.SC2Env(
				  map_name="HellionZerglings",
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


				agent.terminal = False
				agent.s, agent.readout, agent.h_fc1 = agent.createNetwork()

				# define the cost function
				agent.a = tf.placeholder("float", [None, ACTIONS])
				agent.y = tf.placeholder("float", [None])
				agent.readout_action = tf.reduce_sum(tf.multiply(agent.readout, agent.a), reduction_indices=1)
				agent.cost = tf.reduce_mean(tf.square(agent.y - agent.readout_action))
				agent.train_step = tf.train.AdamOptimizer(1e-6).minimize(agent.cost)

				# store the previous observations in replay memory
				agent.D = deque()

				# printing
				agent.a_file = open("logs_" + GAME + "/readout.txt", 'w')
				agent.h_file = open("logs_" + GAME + "/hidden.txt", 'w')

				agent.x_t = agent.grab_screen(timesteps[0])
				print("SHAPE: {}".format(agent.x_t.shape))
				agent.r_t = -0.1
				#agent.ret, agent.x_t = cv2.threshold(agent.x_t, 1, 255, cv2.THRESH_BINARY)
				agent.s_t = np.stack((agent.x_t, agent.x_t, agent.x_t, agent.x_t), axis=2)
				agent.a_t = np.zeros([ACTIONS])
				agent.a_t[0] = 1
				agent.action_index = 0
				agent.total_reward = 0
				agent.readout_t = 0
				agent.sess = tf.InteractiveSession()
				# saving and loading networks
				agent.saver = tf.train.Saver()
				agent.sess.run(tf.initialize_all_variables())
				agent.checkpoint = tf.train.get_checkpoint_state("saved_networks")
				# if agent.checkpoint and agent.checkpoint.model_checkpoint_path:
				#     agent.saver.restore(agent.sess, agent.checkpoint.model_checkpoint_path)
				#     print("Successfully loaded:", agent.checkpoint.model_checkpoint_path)
				# else:
				#     print("Could not find old network weights")

				agent.epsilon = INITIAL_EPSILON
				agent.t = 0

				while True:
				  step_actions = [agent.step(timesteps[0])]
				  timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass
if __name__ == "__main__":
  app.run(main)
