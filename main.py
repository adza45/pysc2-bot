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

os.environ["SC2PATH"] = '/media/adamselement/SAMSUNG SSD/Projects/match-runner/sc2-bot-match-runner/StarCraftII'

#game parameters
ACTIONS = 5 # possible actions: jump, do nothing
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 10000 # timesteps to observe before training
EXPLORE = 300000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.9 # starting value of epsilon
REPLAY_MEMORY = 25000 # number of previous transitions to remember
BATCH = 16 # size of minibatch
FRAME_PER_ACTION = 1
#LEARNING_RATE = 1e-4
LEARNING_RATE = 0.0001
img_rows , img_cols = 64,84
img_channels = 4 #We stack 4 frames

BEACON_REWARD = 30

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_NO_OP = actions.FUNCTIONS.no_op.id

class terranAgent(base_agent.BaseAgent):
	def __init__(self):
		super(terranAgent, self).__init__()
		self.move_coordinates = (0, 0)
		self.previous_score = 0

	def can_do(self, obs, action):
		return action in obs.observation.available_actions

	def grab_screen(self, obs):
		game_data = np.zeros((64, 84, 3), np.uint8)

		for unit in obs.observation.feature_units:
			if unit.unit_type == 317:
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius), (128, 128, 128), math.ceil(int(unit.radius*0.1)))
			elif unit.unit_type == units.Terran.Marine:
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius), (255, 255, 255), math.ceil(int(unit.radius*0.1)))

		#grayed is current state
		image = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)

		resized = cv2.resize(image, dsize=None, fx=2, fy=2)
		cv2.imshow("Map", resized)
		cv2.waitKey(1)

		return image

	def save_obj(self, obj, name ):
		with open('objects/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(self, name ):
		with open('objects/' + name + '.pkl', 'rb') as f:
			return pickle.load(f)

	def step(self, obs, model, observe):
		super(terranAgent, self).step(obs)

		if obs.first():
			self.actions = np.zeros(ACTIONS)
			self.actions[0] =1 #0 => do nothing,

			#self.x_t, self.r_0, self.terminal = game_state.get_state(actions) # get next step after performing the action
			self.x_t = self.grab_screen(obs)
			self.r_t = -0.1
			#self.terminal = False
			self.s_t = np.stack((self.x_t, self.x_t, self.x_t, self.x_t), axis=2)
			self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])  #1*20*40*4

			#self.OBSERVE = OBSERVATION

			if observe:
				self.OBSERVE = 999999999    #We keep observe, never train
				self.epsilon = INITIAL_EPSILON
				print ("Now we load weight")
				model.load_weights("model.h5")
				adam = Adam(lr=LEARNING_RATE)
				model.compile(loss='mse',optimizer=adam)
				print ("Weight load successfully")
			else:                       #We go to training mode
				self.OBSERVE = OBSERVATION
				#self.epsilon = self.load_obj("epsilon")
				model.load_weights("model.h5")
				adam = Adam(lr=LEARNING_RATE)
				model.compile(loss='mse',optimizer=adam)

			#self.t = 0
			self.action_index = 0
			self.Q_sa = 0
			self.loss = 0

			marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine]

			if len(marine) > 0:
			  marine = random.choice(marine)
			  return actions.FUNCTIONS.select_point("select_all_type", (marine.x, marine.y))

		self.score = obs.observation['score_cumulative'][0]
		if self.score > self.previous_score:
			self.r_t += BEACON_REWARD
		else:
			self.r_t -= 0.1

		self.previous_score = self.score

		self.x_t1 = self.grab_screen(obs)
		self.last_time = time.time()

		self.x_t1 = self.x_t1.reshape(1, self.x_t1.shape[0], self.x_t1.shape[1], 1) #1x20x40x1
		self.s_t1 = np.append(self.x_t1, self.s_t[:, :, :, :3], axis=3) # append the new image to input stack and remove the first one

		# store the transition in D
		self.D.append((self.s_t, self.action_index, self.r_t, self.s_t1, self.terminal))

		if len(self.D) > REPLAY_MEMORY:
			self.D.popleft()

		#only train if done observing; sample a minibatch to train on
		#trainBatch(random.sample(self.D, BATCH)) if self.t > self.OBSERVE

		if self.t > self.OBSERVE:
			minibatch = random.sample(self.D, BATCH)

			for i in range(0, len(minibatch)):
				self.loss = 0
				inputs = np.zeros((BATCH, self.s_t.shape[1], self.s_t.shape[2], self.s_t.shape[3]))   #32, 20, 40, 4
				targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2
				state_t = minibatch[i][0]    # 4D stack of images
				action_t = minibatch[i][1]   #self is action index
				reward_t = minibatch[i][2]   #reward at state_t due to action_t
				state_t1 = minibatch[i][3]   #next state
				terminal = minibatch[i][4]   #wheather the agent died or survided due the action
				inputs[i:i + 1] = state_t
				#print(model.predict(state_t))
				targets[i] = model.predict(state_t)  # predicted q values

				self.Q_sa = model.predict(state_t1)      #predict q values for next step

				if terminal:
					targets[i, action_t] = reward_t # if terminated, only equals reward
				else:
					targets[i, action_t] = reward_t + GAMMA * np.max(self.Q_sa)

			self.loss += model.train_on_batch(inputs, targets)

		self.s_t = self.s_t1
		self.t = self.t + 1
		self.total_reward += self.r_t

		# save progress every 1000 iterations
		if self.t % 1000 == 0:
			print("Now we save model")
			#game_state._game.pause() #pause game while saving to filesystem
			model.save_weights("model.h5", overwrite=True)
			self.save_obj(self.D,"D") #saving episodes
			self.save_obj(self.t,"time") #caching time steps
			self.save_obj(self.epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
			with open("model.json", "w") as outfile:
				json.dump(model.to_json(), outfile)

		state = ""
		if self.t <= self.OBSERVE:
			state = "observe"
		elif self.t > self.OBSERVE and self.t <= self.OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print("TIMESTEP", self.t, "/ STATE ", state, "/ EPSILON", self.epsilon, "/ ACTION", self.action_index, "/ REWARD", self.r_t,"/ Q_MAX " , np.max(self.Q_sa), "/ Loss ", self.loss, "/ Total Reward ", self.total_reward)

		self.terminal = False

		self.loss = 0
		self.Q_sa = 0
		self.action_index = 0
		self.r_t = 0 #reward at t
		self.a_t = np.zeros([ACTIONS]) # action at t

		#choose an action epsilon greedy
		if  random.random() <= self.epsilon: #randomly explore an action
			print("----------Random Action----------")
			self.action_index = random.randrange(ACTIONS)
			self.a_t[self.action_index] = 1
		else: # predict the output
			self.q = model.predict(self.s_t)       #input a stack of 4 images, get the prediction
			self.max_Q = np.argmax(self.q)         # chosing index with maximum q value
			self.action_index = self.max_Q
			self.a_t[self.action_index] = 1        # o=> do nothing, 1=> jump

		#We reduced the epsilon (exploration parameter) gradually
		if self.epsilon > FINAL_EPSILON and self.t > self.OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		#run the selected action and observed next state and reward
		#x_t1, r_t, terminal = game_state.get_state(a_t)

		marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine]
		marine = marine[0]


		if self.a_t[0] == 1:
			return actions.FunctionCall(_NO_OP, [])
		elif self.a_t[1] == 1:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (0, marine.y)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])
		elif self.a_t[2] == 1:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (83, marine.y)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])
		elif self.a_t[3] == 1:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (marine.x, 0)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])
		elif self.a_t[4] == 1:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (marine.x, 64)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])

		#return actions.FUNCTIONS.no_op()
		#return actions.FUNCTIONS.no_op()

def buildmodel():
	print("Now we build the model")
	model = Sequential()
	model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4 -> 64, 84, 1
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(5))

	adam = Adam(lr=LEARNING_RATE)
	model.compile(loss='mse',optimizer=adam)
	#create model file if not present
	if not os.path.isfile('model.h5'):
		model.save_weights('model.h5')
	print("We finish building the model")
	return model

def main(unused_argv):
	agent = terranAgent()
	try:
		while True:
			with sc2_env.SC2Env(
				  map_name="MoveToBeacon",
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

				model = buildmodel()
				agent.terminal = False
				#agent.t = 0
				agent.t = agent.load_obj("time") # resume from the previous time step stored in file system
				agent.total_reward = 0
				#agent.D = deque() #load from file system
				agent.D = agent.load_obj("D") #load from file system
				#agent.epsilon = INITIAL_EPSILON
				agent.epsilon = agent.load_obj("epsilon")
				while True:

				  step_actions = [agent.step(timesteps[0], model, False)]

				  if timesteps[0].last():
					  agent.terminal = True
				  timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
  app.run(main)
