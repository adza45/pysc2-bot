from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np
import pandas as pd
import random
import cv2

ACTION_DO_NOTHING = 'donothing'
ACTION_MOVE_LEFT = 'moveleft'
ACTION_MOVE_RIGHT = 'moveright'
ACTION_MOVE_UP = 'moveup'
ACTION_MOVE_DOWN = 'movedown'

smart_actions = [
	ACTION_DO_NOTHING,
	ACTION_MOVE_LEFT,
	ACTION_MOVE_RIGHT,
	ACTION_MOVE_UP,
	ACTION_MOVE_DOWN
]

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

_NOT_QUEUED = [0]
_QUEUED = [1]

#BEACON_REWARD = 0.2
BEACON_REWARD = 1

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow


class QLearningTable:
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, observation):
		self.check_state_exist(observation)

		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.ix[observation, :]

			# some actions have the same value
			state_action = state_action.reindex(
				np.random.permutation(state_action.index))

			action = state_action.idxmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)

		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		self.check_state_exist(s)

		q_predict = self.q_table.ix[s, a]
		q_target = r + self.gamma * self.q_table.ix[s_, :].max()

		# update
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class terranAgent(base_agent.BaseAgent):
	def __init__(self):
		super(terranAgent, self).__init__()

		self.move_coordinates = (0, 0)

		self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

		self.previous_player_score = 0

		self.previous_action = None
		self.previous_state = None

	def can_do(self, obs, action):
		return action in obs.observation.available_actions

	def step(self, obs):
		super(terranAgent, self).step(obs)
		if obs.first():
			print(self.qlearn.q_table)
			print("First Obs")
			marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine]

			if len(marine) > 0:
			  marine = random.choice(marine)
			  return actions.FUNCTIONS.select_point("select_all_type", (marine.x, marine.y))


		game_data = np.zeros((64, 84, 3), np.uint8)

		for unit in obs.observation.feature_units:
			# print("Unit position: {}, {}".format(unit.x, unit.y))
			# print("Unit Type: {}".format(unit.unit_type))
			# cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))
			if unit.unit_type == units.Terran.Marine:
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius), (255, 255, 255), -1)
			elif unit.unit_type == 317:
				cv2.circle(game_data, (int(unit.x), int(unit.y)), int(unit.radius), (128, 128, 128), -1)

		#grayed is current state
		grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)

		resized = cv2.resize(grayed, dsize=None, fx=2, fy=2)

		cv2.imshow("Map", resized)
		cv2.waitKey(1)

		player_score = obs.observation['score_cumulative'][0]
		#killed_building_score = obs.observation['score_cumulative'][6]

		#print("score : {}".format(obs.observation['score_cumulative'][0]))

		current_state = grayed
		# current_state = [
		#     supply_depot_count,
		#     barracks_count,
		#     supply_limit,
		#     army_supply,
		# ]

		if self.previous_action is not None:
			reward = 0

			if player_score > self.previous_player_score:
				reward += BEACON_REWARD
				print("Found beacon, new reward: {}".format(reward))
			else:
				reward -= .001

			self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

		rl_action = self.qlearn.choose_action(str(current_state))
		smart_action = smart_actions[rl_action]

		#print("Smart_action: {}".format(smart_action))

		self.previous_player_score = player_score
		self.previous_state = current_state
		self.previous_action = rl_action


		marine = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Marine]
		marine = marine[0]

		# smart_action = smart_actions[random.randrange(0, len(smart_actions))]

		if smart_action == ACTION_DO_NOTHING:
			return actions.FunctionCall(_NO_OP, [])
		elif smart_action == ACTION_MOVE_LEFT:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (0, marine.y)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])
		elif smart_action == ACTION_MOVE_RIGHT:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (83, marine.y)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])
		elif smart_action == ACTION_MOVE_UP:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (marine.x, 0)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])
		elif smart_action == ACTION_MOVE_DOWN:
			if self.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
				self.move_coordinates = (marine.x, 64)
				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.move_coordinates])

		return actions.FUNCTIONS.no_op()

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

				while True:

				  step_actions = [agent.step(timesteps[0])]

				  if timesteps[0].last():
					  pass
				  timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
  app.run(main)
