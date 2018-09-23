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
OBSERVE = 10000 # timesteps to observe before training
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
        W_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        W_fc1 = weight_variable([1600, 512])
        b_fc1 = bias_variable([512])

        W_fc2 = weight_variable([512, ACTIONS])
        b_fc2 = bias_variable([ACTIONS])

        # input layer
        s = tf.placeholder("float", [None, 64, 84, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        return s, readout, h_fc1

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

	def step(self, obs):
		super(terranAgent, self).step(obs)

		if obs.first():

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
        self.ret, self.x_t1 = cv2.threshold(self.x_t1, 1, 255, cv2.THRESH_BINARY)
        self.x_t1 = np.reshape(self.x_t1, (64, 84, 1))
        self.s_t1 = np.append(self.x_t1, self.s_t[:, :, :3], axis=2)

		self.last_time = time.time()

		# store the transition in D
		self.D.append((self.s_t, self.a_t, self.r_t, self.s_t1, self.terminal))

		if len(self.D) > REPLAY_MEMORY:
			self.D.popleft()

		#only train if done observing; sample a minibatch to train on
		#trainBatch(random.sample(self.D, BATCH)) if self.t > self.OBSERVE

		if self.t > self.OBSERVE:
            minibatch = random.sample(self.D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {self.s : s_j1_batch})
            for i in range(0, len(minibatch)):
                self.terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
		self.s_t = self.s_t1
		self.t = self.t + 1
		self.total_reward += self.r_t

		# save progress every 1000 iterations
        # save progress every 10000 iterations
        if self.t % 10000 == 0:
            saver.save(self.sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

		state = ""
		if self.t <= OBSERVE:
			state = "observe"
		elif self.t > OBSERVE and self.t <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

        print("TIMESTEP", self.t, "TOTAL REWARD", self.total_reward, "/ STATE", self.state, \
            "/ EPSILON", self.epsilon, "/ ACTION", self.action_index, "/ REWARD", self.r_t, \
            "/ Q_MAX %e" % np.max(self.readout_t))

		self.terminal = False

        # choose an action epsilon greedily
        self.readout_t = readout.eval(feed_dict={self.s : [self.s_t]})[0]
        self.a_t = np.zeros([ACTIONS])
        self.action_index = 0
        if self.t % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                print("----------Random Action----------")
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

    			agent.x_t = self.grab_screen(obs)
    			agent.r_t = -0.1
                agent.ret, self.x_t = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
                agent.s_t = np.stack((agent.x_t, agent.x_t, agent.x_t, agent.x_t), axis=2)

                # saving and loading networks
                saver = tf.train.Saver()
                agent.sess.run(tf.initialize_all_variables())
                checkpoint = tf.train.get_checkpoint_state("saved_networks")
                # if checkpoint and checkpoint.model_checkpoint_path:
                #     saver.restore(sess, checkpoint.model_checkpoint_path)
                #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
                # else:
                #     print("Could not find old network weights")

                self.epsilon = INITIAL_EPSILON
                self.t = 0

				while True:

				  step_actions = [agent.step(timesteps[0])]

				  if timesteps[0].last():
					  agent.terminal = True
				  timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
  app.run(main)
