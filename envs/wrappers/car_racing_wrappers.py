# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque

import numpy as np
import torch
import gym

from .vec_env import VecEnvWrapper


class CarRacingWrapper(gym.Wrapper):
	def __init__(self, 
				 env, 
				 grayscale=True, 
				 reward_shaping=True,
				 sparse_rewards=False,
				 early_termination=True, 
				 timelimit_bonus=True,
				 num_action_repeat=8,
				 nstack=1,
				 channel_first=False,
				 crop=True,
				 eval_=False):
		super().__init__(env)

		self.eval_ = eval_

		self.grayscale = grayscale
		self.reward_shaping = reward_shaping
		self.sparse_rewards = sparse_rewards
		self.num_action_repeat = num_action_repeat
		self.early_termination = early_termination
		self.timelimit_bonus = timelimit_bonus
		self.crop = crop

		self.nstack = nstack
		self.channel_first = channel_first

		self.reset_reward_history()

		self.set_observation_space()

		if self.sparse_rewards:
			self.accumulated_rewards = 0.0

	def reset_reward_history(self):
		if self.early_termination:
			self.reward_history = deque([0]*100,maxlen=100)

		if self.sparse_rewards:
			self.accumulated_rewards = 0.0

	def _preprocess(self, obs):
		# Crop
		if self.crop:
			obs = obs[:-12, 6:-6]

		# Grayscale
		if self.grayscale:
			obs = np.expand_dims(np.dot(obs[..., :], [0.299, 0.587, 0.114]), -1)
		
		obs = obs/128. - 1.

		return obs

	@property
	def _average_reward(self):
		return np.mean(self.reward_history)

	def _reset_stack(self, obs):
		if self.nstack > 1:
			self.stack = [obs] * self.nstack # four frames for decision
			obs = np.concatenate(self.stack, axis=-1)
		return obs

	def _transpose(self, obs):
		if self.channel_first:
			obs = np.swapaxes(obs, 0, 2)
			obs = np.swapaxes(obs, 1, 2)
		return obs

	# Public interface
	def reset(self):
		self.reset_reward_history()

		if self.eval_:
			obs = self.env.reset()
			obs = self._preprocess(obs)

			obs = self._reset_stack(obs)

			obs = self._transpose(obs)

			return obs
		else:
			return self.env.reset()

	def reset_random(self):
		self.reset_reward_history()

		obs = self.env.reset_random()

		obs = self._preprocess(obs)
		obs = self._reset_stack(obs)
		obs = self._transpose(obs)

		return obs

	def reset_agent(self):
		self.reset_reward_history()

		obs = self.env.reset_agent()

		obs = self._preprocess(obs)
		obs = self._reset_stack(obs)
		obs = self._transpose(obs)

		return obs

	def reset_to_level(self, level):
		self.reset_reward_history()

		obs = self.env.reset_to_level(level)

		obs = self._preprocess(obs)
		obs = self._reset_stack(obs)
		obs = self._transpose(obs)

		return obs

	def step(self, action):
		done = False
		total_reward = 0
		for i in range(self.num_action_repeat):
			obs, reward, die, info = self.env.step(action)

			if self.reward_shaping:
				# Don't penalize "done state"
				if die:
					if self.timelimit_bonus:
						reward += 100

				# Green penalty
				if np.mean(obs[:, :, 1]) > 185.0:
					reward -= 0.05

				if self.early_termination:
					self.reward_history.append(reward)
					done = True if self._average_reward <= -0.1 else False

			total_reward += reward
			# If no reward recently, end the episode
			if done or die:
				break

		obs = self._preprocess(obs)

		if self.nstack > 1:
			self.stack.pop(0)
			self.stack.append(obs)
			obs = np.concatenate(self.stack, axis=-1)

		obs = self._transpose(obs)

		if self.sparse_rewards:
			if self.env.goal_reached:
				revealed_reward = self.accumulated_rewards
				self.accumulated_rewards = 0.0
			else:
				self.accumulated_rewards += total_reward
				revealed_reward = 0.0
		else:
			revealed_reward = total_reward

		# obs = np.expand_dims(obs, 0)
		return obs, revealed_reward, done or die, info

	def set_observation_space(self):
		obs_space = self.env.observation_space
		num_channels = 1 if self.grayscale else 3

		if self.nstack > 1:
			num_channels *= self.nstack

		# Cropped and potentially grayscaled observation
		if self.crop:
			obs_shape = (obs_space.shape[0] - 12, obs_space.shape[1] - 12, num_channels)
		else:
			obs_shape = (obs_space.shape[0], obs_space.shape[1], num_channels)

		if self.channel_first:
			obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

		self.observation_space = gym.spaces.Box(
			low=-1,
			high=1,
			shape=obs_shape,
			dtype='float32')

		return self.observation_space
