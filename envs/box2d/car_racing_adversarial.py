# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import math
import random

import numpy as np
import gym
from gym.envs.box2d.car_dynamics import Car
from envs.registration import register as gym_register

from .car_racing_bezier import CarRacingBezier
from envs.box2d import *

class CarRacingBezierAdversarial(CarRacingBezier):
	def __init__(self, 
		n_control_points=12, 
		random_z_dim=4, 
		track_name=None, 
		bezier=True,
		show_borders=True,
		show_indicators=True, 
		birdseye=False, 
		seed=None,
		fixed_environment=False, 
		animate_zoom=False,
		min_rad_ratio=None,
		max_rad_ratio=None,
		use_sketch=None,
		choose_start_pos=False,
		use_categorical=False,
		clip_reward=None,
    sparse_rewards=False,
		num_goal_bins=24,
		verbose=1):

		super().__init__(
			track_name=track_name, 
			bezier=bezier,
			show_borders=show_borders,
			show_indicators=show_indicators,
			birdseye=birdseye,
			seed=seed,
			fixed_environment=fixed_environment,
			animate_zoom=False,
			clip_reward=clip_reward,
        	sparse_rewards=sparse_rewards,
        	num_goal_bins=num_goal_bins,
			verbose=verbose)
		
		self.passable = True

		self.random_z_dim = random_z_dim

		self.choose_start_pos = choose_start_pos
		self._adv_start_alpha = None

		self.n_control_points = n_control_points
		self._adversary_control_points = []
		# sketch_dim = int(np.round(self.playfield/self.track_width))
		sketch_dim = 10
		self.sketch_dim = sketch_dim # Should be 50
		self.sketch_ratio = self.playfield/self.sketch_dim
		self._adversary_sketch = np.zeros((self.sketch_dim, self.sketch_dim))

		self.adversary_max_steps = n_control_points
		if choose_start_pos:
			self.adversary_max_steps += 1 # Extra step to choose start pos
		if sparse_rewards:
			self.adversary_max_steps += 1 # Extra step to choose goal bin (last action)

		self.adversary_step_count = 0

		# === Adversary env observations ===
		self._clear_adversary_sketch()
		self._adversary_sketch_dirty = False

		# === Adversary observation and action space ===
		self.adversary_ts_obs_space = gym.spaces.Box(
			low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
		self.adversary_randomz_obs_space = gym.spaces.Box(
			low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
		self.adversary_control_points_obs_space = gym.spaces.Box(
			low=0,
			high=sketch_dim,
			shape=(1, sketch_dim, sketch_dim),
			dtype='uint8')

		if sparse_rewards:
			self.adversary_goal_bin_obs_space = gym.spaces.Box(
					low=0,
					high=num_goal_bins + 1, # +1 for placeholder (at last, extra index)
					shape=(1,),
					dtype='uint8'
				)
			self.adversary_observation_space = gym.spaces.Dict(
				{'control_points': self.adversary_control_points_obs_space,
				 'time_step': self.adversary_ts_obs_space,
				 'random_z': self.adversary_randomz_obs_space,
				 'goal_bin': self.adversary_goal_bin_obs_space})
		else:
			self.adversary_observation_space = gym.spaces.Dict(
				{'control_points': self.adversary_control_points_obs_space,
				 'time_step': self.adversary_ts_obs_space,
				 'random_z': self.adversary_randomz_obs_space})

		# Note adversary_action_space is only used to communicate to storage
		# the proper dimensions for storing the *unprocessed* actions
		if use_categorical:
			action_low, action_high = np.array((0,)), np.array((self.sketch_dim**2 + 1,)) # +1 for skip action
			if sparse_rewards:
				action_low = np.array((0, *action_low))
				action_high = np.array((1, *action_high))
			self.adversary_action_space = gym.spaces.Box(
				low=action_low, high=action_high, dtype='uint8')
		else:
			action_shape = (3,)
			if sparse_rewards:
				action_shape = (4,) # First dim stores goal flag
			self.adversary_action_space = gym.spaces.Box(
				low=0, high=1, shape=action_shape, dtype='float32')

	@property
	def processed_action_dim(self):
		return 3
	
	def reset(self):
		self.steps = 0
		self.adversary_step_count = 0

		if self._adversary_sketch_dirty:
			self._clear_adversary_sketch()

		# Clear track and agent status
		self.reset_agent_status()

		obs = {
			'control_points': np.expand_dims(self._adversary_sketch,0),
			'time_step': [self.adversary_step_count],
			'random_z': self.generate_random_z()
		}

		# Set goal bin to 1 more than max 0-indexed goal bin
		if self.sparse_rewards:
			obs.update({'goal_bin': [self.num_goal_bins]}) 
			self.goal_bin = None

		return obs

	def _alpha_from_xy(self, x,y):
		alpha = np.arctan2(y,x)
		if alpha < 0:
			alpha += 2*math.pi

		return alpha

	def _set_start_position(self, x, y):
		_,_,unnorm_x,unnorm_y = self.unnormalize_xy(x,y)
		u = np.mean(np.array(self._adversary_control_points), axis=0)

		alpha = self._alpha_from_xy(unnorm_x-u[0],unnorm_y-u[1])

		self._adv_start_alpha = alpha

		return alpha

	def _closest_track_index(self, alpha):
		if len(self._adversary_control_points) == 0:
			return 0

		u = np.mean(np.array(self._adversary_control_points), axis=0)
		track_alphas = np.array([self._alpha_from_xy(x-u[0],y-u[1]) for _,_,x,y in self.track])

		i = np.argmin(np.abs(track_alphas - alpha))

		return np.argmin(np.abs(track_alphas - alpha))

	def reset_agent_status(self):
		# Reset env-specific meta-data
		self._destroy()
		self.reward = 0.0
		self.prev_reward = 0.0
		self.tile_visited_count = 0
		self.t = 0.0
		self.road_poly = []
		
		self.steps = 0
		
		self._create_track(control_points=self.track_data)

		if self._adv_start_alpha is None:
			start_idx = 0
		else:
			start_idx = self._closest_track_index(self._adv_start_alpha)

		beta0, x0, y0 = 0,0,0
		if self._adversary_sketch_dirty: # Car only if track (because reset was called)
			beta0, x0, y0 = self.track[start_idx][1:4]
			x0 -= self.x_offset
			y0 -= self.y_offset

		if self.car:
			self.car.destroy()
			self.car = None
		self.car = Car(self.world, beta0, x0, y0)

		self.reset_sparse_state()

	def reset_agent(self):
		self.reset_agent_status()
		return self.step(None)[0]

	def reset_to_level(self, level):
		self.reset()
		level_features = eval(level)
		self._adversary_control_points = level_features[:-1]
		self._adv_start_alpha = level_features[-1]

		# Build new level
		self._adversary_sketch_dirty = True
		self._create_track_adversary()
		obs = self.reset_agent()

		return obs

	@property
	def level(self):
		return str(tuple(self._adversary_control_points + [self._adv_start_alpha,]))

	def generate_random_z(self):
		return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

	def unnormalize_xy(self, x,y):
		scaled_x = int(np.minimum(np.maximum(np.round(self.sketch_dim*x), 0), self.sketch_dim - 1))
		scaled_y = int(np.minimum(np.maximum(np.round(self.sketch_dim*y), 0), self.sketch_dim - 1))

		unnorm_x = (scaled_x + 1)*self.sketch_ratio
		unnorm_y = (scaled_y + 1)*self.sketch_ratio

		return scaled_x, scaled_y, unnorm_x, unnorm_y

	def _update_adversary_sketch(self, x, y):
		# Update sketch based on latest control points
		scaled_x, scaled_y, unnorm_x, unnorm_y = self.unnormalize_xy(x,y)

		self._adversary_control_points.append((unnorm_x, unnorm_y))
		self._adversary_sketch_dirty = True

		self._adversary_sketch[scaled_x][scaled_y] = 1.0

		return unnorm_x, unnorm_y, self._adversary_sketch

	def _clear_adversary_sketch(self):
		self._adversary_sketch.fill(0)
		self._adversary_control_points = []
		self._adversary_sketch_dirty = False
		self._adv_start_alpha = None

	def _create_track_adversary(self):
		# Compile adversary control points into playfield coordinates
		# Note that each sketch grid point corresponds to to at least a track width apart
		if self.bezier:
			self.track_data = self._adversary_control_points
		else:
			raise NotImplementedError	

	@property
	def is_goal_step(self):
		if self.sparse_rewards:
			return self.adversary_step_count == self.adversary_max_steps - 1
		else:
			return False
	
	@property
	def is_start_pos_step(self):
		if self.choose_start_pos:
			return self.adversary_step_count == self.n_control_points
		else:
			return False

	def step_adversary(self, action):
		# Updates sketch with a new control pt (action)
		# Obs is the latest sketch of control points scaled by self.sketch_dim.
		done = False

		goal_bin = self.num_goal_bins
		if self.is_goal_step:
			goal_bin = action
		else:
			x,y,skip = action

		# Place control point
		if self.adversary_step_count < self.n_control_points:
			if not (self.adversary_step_count > 3 and np.isclose(skip, 1)):
				self._update_adversary_sketch(x,y)
		elif self.is_start_pos_step:
			self._set_start_position(x,y)
		elif self.is_goal_step:
			self.goal_bin = goal_bin
			self.set_goal(goal_bin)

		self.adversary_step_count += 1

		if self.adversary_step_count == self.adversary_max_steps:
			self._create_track_adversary()
			self.reset_agent_status()
			done = True

		obs = {
			'control_points': np.expand_dims(self._adversary_sketch,0), # 1 x sketch_dim x sketch_dim
			'time_step': [self.adversary_step_count],
			'random_z': self.generate_random_z()
		}

		if self.sparse_rewards:
			obs.update({'goal_bin': [goal_bin]}) 

		return obs, 0., done, {}

	def reset_random(self):
		self._adversary_sketch_dirty = True

		if self.fixed_environment:
			self.seed(self.level_seed)

		if self.sparse_rewards:
			self.goal_bin = None
			self.set_goal()

		return super().reset()


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


gym_register(
	id='CarRacing-Bezier-Adversarial-v0', 
	entry_point=module_path + ':CarRacingBezierAdversarial',
    max_episode_steps=1000,
    reward_threshold=900)
