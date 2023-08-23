# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from .common import *
from .distributions import FixedCategorical, Categorical
from .popart import PopArt


class CarRacingNetwork(DeviceAwareModule):
	def __init__(self, 
		obs_shape,
		action_space,
		hidden_size=100,
		crop=False,
		use_popart=False):      
		super(CarRacingNetwork, self).__init__()

		m = obs_shape[-2] # x input dim
		n = obs_shape[-1] # y input dim
		c = obs_shape[-3] # channel input dim

		self.action_low = np.array(action_space.low, dtype=np.float32)
		self.action_high = np.array(action_space.high, dtype=np.float32)
		self.action_dim = action_space.shape[0]

		self.crop = crop

		if self.crop:
			self.image_embed = nn.Sequential(  # input shape (4, 84, 84)
				nn.Conv2d(c, 8, kernel_size=2, stride=2),
				nn.ReLU(),  # activation
				nn.Conv2d(8, 16, kernel_size=2, stride=2),  # (8, 42, 42)
				nn.ReLU(),  # activation
				nn.Conv2d(16, 32, kernel_size=2, stride=2),  # (16, 21, 21)
				nn.ReLU(),  # activation
				nn.Conv2d(32, 64, kernel_size=2, stride=2),  # (32, 10, 10)
				nn.ReLU(),  # activation
				nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
				nn.ReLU(),  # activation
				nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
				nn.ReLU(),  # activation
			)  # output shape (256, 1, 1)
		else:
			self.image_embed = nn.Sequential(  # input shape (4, 96, 96)
				nn.Conv2d(c, 8, kernel_size=4, stride=2),
				nn.ReLU(),  # activation
				nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
				nn.ReLU(),  # activation
				nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
				nn.ReLU(),  # activation
				nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
				nn.ReLU(),  # activation
				nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
				nn.ReLU(),  # activation
				nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
				nn.ReLU(),  # activation
			)  # output shape (256, 1, 1)

		self.image_embedding_size = 256

		# # Policy head
		self.actor_fc = nn.Sequential(
			init_relu_(nn.Linear(self.image_embedding_size, hidden_size)),
			nn.ReLU(),
		)
		self.actor_alpha = nn.Sequential(
			init_relu_(nn.Linear(hidden_size, self.action_dim)),
			nn.Softplus(),
		)
		self.actor_beta = nn.Sequential(
			init_relu_(nn.Linear(hidden_size, self.action_dim)),
			nn.Softplus(),
		)

		# Value head
		if use_popart:
			value_out = init_(PopArt(hidden_size, 1))
			self.popart = value_out
		else:
			value_out = init_(nn.Linear(hidden_size, 1))
			self.popart = None

		self.critic = nn.Sequential(
			init_relu_(nn.Linear(self.image_embedding_size, hidden_size)),
			nn.ReLU(),
			value_out
		)

		# apply_init_(self.modules(), gain=nn.init.calculate_gain('relu'))
		self.apply(self._weights_init)

		self.train()

	@staticmethod
	def _weights_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
			nn.init.constant_(m.bias, 0.1)

	@property
	def is_recurrent(self):
		return False

	@property
	def recurrent_hidden_state_size(self):
		# """Size of rnn_hx."""
		return 1

	def forward(self, inputs, rnn_hxs, masks):
		raise NotImplementedError

	def process_action(self, action):
		return action*(self.action_high - self.action_low) + self.action_low

	def act(self, inputs, rnn_hxs, masks, deterministic=False):
		image_embedding = self.image_embed(inputs).squeeze()

		actor_fc_embed = self.actor_fc(image_embedding)
		alpha = 1 + self.actor_alpha(actor_fc_embed)
		beta = 1 + self.actor_beta(actor_fc_embed)

		dist = Beta(alpha, beta)
		# action = alpha/(alpha + beta)
		action = dist.sample()
		# For continuous action spaces, we just return dirac delta over 
		# sampled action tuple
		action_log_dist = dist.log_prob(action).sum(dim=-1).unsqueeze(-1)

		value = self.critic(image_embedding)

		if inputs.shape[0] == 1:
			action = action.unsqueeze(0)

		return value, action, action_log_dist, rnn_hxs

	def get_value(self, inputs, rnn_hxs, masks):
		image_embedding = self.image_embed(inputs).squeeze()
		return self.critic(image_embedding)

	def evaluate_actions(self, inputs, rnn_hxs, masks, action, return_policy_logits=False):
		image_embedding = self.image_embed(inputs).squeeze()

		actor_fc_embed = self.actor_fc(image_embedding)
		alpha = self.actor_alpha(actor_fc_embed) + 1
		beta = self.actor_beta(actor_fc_embed) + 1

		a_range = (torch.min(alpha).item(), torch.max(alpha).item())
		b_range = (torch.min(beta).item(), torch.max(beta).item())

		dist = Beta(alpha, beta)
		action_log_probs = dist.log_prob(action).sum(dim=-1).unsqueeze(-1)
		dist_entropy = dist.entropy().mean()

		value = self.critic(image_embedding)

		if return_policy_logits:
			return value, action_log_probs, dist_entropy, rnn_hxs, dist

		return value, action_log_probs, dist_entropy, rnn_hxs


class CarRacingBezierAdversaryEnvNetwork(DeviceAwareModule):
	def __init__(self, 
		observation_space,
		action_space,
		scalar_fc=8,
		use_categorical=False,
		use_skip=False,
		choose_start_pos=False,
		use_popart=False,
		set_start_pos=False,
		use_goal=False,
		num_goal_bins=1):  
		super().__init__()

		self.sketch_dim = observation_space['control_points'].shape
		self.random_z_dim = observation_space['random_z'].shape[0]
		self.total_time_steps = observation_space['time_step'].high[0]
		self.time_step_dim = self.total_time_steps + 1 # Handles terminal time step

		if use_goal:
			self.goal_bin_dim = observation_space['goal_bin'].high[0]

		self.scalar_fc = scalar_fc

		self.set_start_pos = set_start_pos
		self.use_categorical = use_categorical
		self.use_skip = use_skip
		self.use_goal = use_goal
		self.num_goal_bins = num_goal_bins

		self.random = False
		self.action_space = action_space

		self.n_control_points = self.time_step_dim
		if self.use_goal:
			self.n_control_points -= 1
		if self.set_start_pos:
			self.n_control_points -= 1

		if use_categorical:
			self.action_dim = np.prod(self.sketch_dim) + 1 # +1 for skip action
		else:
			self.action_dim = action_space.shape[0]
			if self.use_goal:
				self.action_dim -= 1 # Sinc we don't learn Beta for goal prefix

		self.sketch_embedding = nn.Sequential(
			Conv2d_tf(1, 8, kernel_size=2, stride=1, padding='valid'), # (8, 9, 9)
			Conv2d_tf(8, 16, kernel_size=2, stride=1, padding='valid'), # (16, 8, 8)
			nn.Flatten(),
			nn.ReLU()  # output is 1024 dimensions
		)
		self.sketch_embed_dim = 1024

		# Time step embedding
		self.ts_embedding = nn.Linear(self.time_step_dim, scalar_fc)

		self.base_output_size = self.sketch_embed_dim + \
			self.scalar_fc + self.random_z_dim

		if use_goal:
			self.goal_bin_embedding = nn.Linear(num_goal_bins + 1, self.scalar_fc)
			self.base_output_size += self.scalar_fc

		# Value head
		self.critic = init_(nn.Linear(self.base_output_size, 1))

		# Value head
		if use_popart:
			self.critic = init_(PopArt(self.base_output_size, 1))
			self.popart = self.critic
		else:
			self.critic = init_(nn.Linear(self.base_output_size, 1))
			self.popart = None

		# Policy heads
		if self.use_categorical:
			self.actor = nn.Sequential(
				init_(nn.Linear(self.base_output_size, 256)),
				nn.ReLU(),
				init_(nn.Linear(256, self.action_dim)),
			)	
		else:
			self.fc_alpha = nn.Sequential(
				init_relu_(nn.Linear(self.base_output_size, self.action_dim)),
				nn.Softplus()
			)
			self.fc_beta = nn.Sequential(
				init_relu_(nn.Linear(self.base_output_size, self.action_dim)),
				nn.Softplus()
			)

		# Create a policy head to select a goal bin
		if use_goal:
			self.goal_head = nn.Sequential(
				init_(nn.Linear(self.base_output_size, 256)),
				nn.ReLU(),
				init_(nn.Linear(256, num_goal_bins)),
			)	

		apply_init_(self.modules())

		self.train()

	@property
	def is_recurrent(self):
		return False

	@property
	def recurrent_hidden_state_size(self):
		# """Size of rnn_hx."""
		return 1

	def forward(self, inputs, rnn_hxs, masks):
		raise NotImplementedError

	def _forward_base(self, inputs, rnn_hxs, masks):
		sketch = inputs['control_points']
		time_step = inputs['time_step']
		in_z = inputs['random_z']

		in_sketch = self.sketch_embedding(sketch)

		in_time_step = one_hot(self.time_step_dim, time_step).to(self.device)
		in_time_step = self.ts_embedding(in_time_step)

		if self.use_goal:
			goal_bin = inputs['goal_bin']
			in_goal_bin = one_hot(self.goal_bin_dim, goal_bin).to(self.device)
			in_goal_bin = self.goal_bin_embedding(in_goal_bin)
			in_embedded = torch.cat((in_sketch, in_time_step, in_z, in_goal_bin), dim=-1)
		else:
			in_embedded = torch.cat((in_sketch, in_time_step, in_z), dim=-1)

		return in_embedded

	def process_action(self, action):
		if self.use_goal:
			if action[0][0]: # Check if it's a goal step
				action_ = action[:,1]
				return action_
			else:
				action = action[:,1:]

		if self.use_categorical:
			x = ((action - 1.) % self.sketch_dim[-1])/self.sketch_dim[-1]
			y = ((action - 1.) // self.sketch_dim[-2])/self.sketch_dim[-2]

			skip_action = (action == 0).float() 

			action_ = torch.cat((x,y,skip_action), dim=-1)
		else:
			xy_action = action[:,:-1]
			skip_logits = torch.log(action[:,-1] + 1e-5) # ensure > 0
			skip_action = F.gumbel_softmax(skip_logits, tau=1, hard=True).unsqueeze(-1)

			action_ = torch.cat((xy_action, skip_action), dim=-1)

		return action_

	def _sketch_to_mask(self, sketch):
		mask = torch.cat((torch.zeros(sketch.shape[0],1, dtype=torch.bool, device=self.device), 
						 (sketch.flatten(1).bool())), 
						 dim=-1)
		return mask

	def _is_goal_step(self, t):
		if self.use_goal:
			return t == self.total_time_steps - 1 # since time_step is +1 total time steps
		else: 
			return False

	def _is_start_pos_step(self, t):
		if self.set_start_pos:
			return t == self.n_control_points
		else:
			return False

	def act_random(self, inputs, rnn_hxs):
		time_step = inputs['time_step'][0]
		is_goal_step = self._is_goal_step(time_step)

		B = inputs['time_step'].shape[0]
		values = torch.zeros((B,1), device=self.device)
		action_log_dist = torch.ones((B,1), device=self.device)

		action_shape = self.action_space.shape
		if self.use_goal:
			action_shape = (action_shape[0] - 1,)

		# import pdb; pdb.set_trace()
		if is_goal_step:
			# random goal bin
			action = torch.zeros((B,1), dtype=torch.int64, device=self.device)
			for b in range(B):
				action[b] = np.random.randint(self.num_goal_bins)
		else:
			action = torch.zeros(B, *action_shape, 
					dtype=torch.int64, device=self.device)
			if self.use_categorical:
				action_high = self.action_space.high[1] if self.use_goal \
					else self.action_space.high[0]
				for b in range(B):
					action[b] = np.random.randint(1, action_high) # avoid skip action 0
			else:
				for b in range(B):
					action[b] = np.random.rand(self.action_shape)

		if self.use_goal:
			if is_goal_step:
				prefix = torch.ones_like(action[:,0].unsqueeze(-1))
			else:
				prefix = torch.zeros_like(action[:,0].unsqueeze(-1))

			action = torch.cat((prefix, action), dim=-1)

		return values, action, action_log_dist, rnn_hxs

	def act(self, inputs, rnn_hxs, masks, deterministic=False):
		if self.random:
			return self.act_random(inputs, rnn_hxs)

		in_embedded = self._forward_base(inputs, rnn_hxs, masks)

		value = self.critic(in_embedded)

		time_step = inputs['time_step'][0]
		is_goal_step = self._is_goal_step(time_step)

		if is_goal_step:
			# generate goal bin action
			logits = self.goal_head(in_embedded)

			dist = FixedCategorical(logits=logits)
			action = dist.sample()

			action_log_probs = dist.log_probs(action)
		else:
			if self.use_categorical:
				logits = self.actor(in_embedded)
				mask = self._sketch_to_mask(inputs['control_points'])

				# Conditionally mask out skip action
				if self.use_skip:
					if self._is_start_pos_step(time_step):
						mask[:,0] = True # Can't skip setting start pos if necessary
					else:
						mask[mask.sum(-1) < 3,0] = True  
				else:
					mask[:,0] = True

				logits[mask] = torch.finfo(logits.dtype).min
				
				dist = FixedCategorical(logits=logits)
				action = dist.sample()

				action_log_probs = dist.log_probs(action)
			else:
				# All B x 3
				alpha = 1 + self.fc_alpha(in_embedded)
				beta = 1 + self.fc_beta(in_embedded)

				dist = Beta(alpha, beta)
				action = dist.sample()

				action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)

		# Hack: Just set action log dist to action log probs, since it's not used.
		action_log_dist = action_log_probs

		# Append [0] or [1] prefix to actions to signal goal step
		if self.use_goal:
			if is_goal_step:
				prefix = torch.ones_like(action[:,0].unsqueeze(-1))
			else:
				prefix = torch.zeros_like(action[:,0].unsqueeze(-1))
			action = torch.cat((prefix, action), dim=-1)

		return value, action, action_log_dist, rnn_hxs

	def get_value(self, inputs, rnn_hxs, masks):
		in_embedded = self._forward_base(inputs, rnn_hxs, masks)

		return self.critic(in_embedded)

	def evaluate_actions(self, inputs, rnn_hxs, masks, action):
		B = len(inputs['time_step'])
		in_embedded = self._forward_base(inputs, rnn_hxs, masks)

		value = self.critic(in_embedded)

		time_steps = inputs['time_step']
		mask = self._sketch_to_mask(inputs['control_points'])

		if self.use_goal:
			action = action[:,1:]

		# Need to mask out both selectively
		goal_steps = self._is_goal_step(time_steps)
		if self.use_goal:
			goal_steps = goal_steps.flatten()
			has_goal_steps = goal_steps.any()
			has_nongoal_steps = (~goal_steps).any()
		else:
			has_goal_steps = False
			has_nongoal_steps = True

		if has_goal_steps:
			# Get logits for goal actions
			goal_in_embed = in_embedded[goal_steps]
			action_in_embed = in_embedded[~goal_steps]
			mask = mask[~goal_steps]

			goal_actions = action[goal_steps][0]
			goal_logits = self.goal_head(goal_in_embed)

			action = action[~goal_steps]

			goal_dist = FixedCategorical(logits=goal_logits)
			goal_action_log_probs = goal_dist.log_probs(goal_actions)
		else:
			action_in_embed = in_embedded

		if has_nongoal_steps:
			if self.use_categorical:
				logits = self.actor(action_in_embed)

				if self.use_skip:
					start_pos_steps = self._is_start_pos_step(time_steps[~goal_steps])
					mask[mask.sum(-1) < 3,0] = True
					logits[start_pos_steps] = torch.finfo(logits.dtype).min
				else:
					mask[:,0] = True

				logits[mask] = torch.finfo(logits.dtype).min
				
				dist = FixedCategorical(logits=logits)

				action_log_probs = dist.log_probs(action)
			else:
				# All B x 3
				alpha = 1 + self.fc_alpha(action_in_embed)
				beta = 1 + self.fc_beta(action_in_embed)

				dist = Beta(alpha, beta)

				action_log_probs = dist.log_prob(action).sum(dim=1).unsqueeze(1)

		if self.use_goal:
			combined_log_probs = torch.zeros((B,1), dtype=torch.float, device=self.device)
			mean_entropy = 0
			if goal_steps.any():
				combined_log_probs[goal_steps] = goal_action_log_probs
				mean_entropy += goal_dist.entropy().sum()
			if (~goal_steps).any():
				combined_log_probs[~goal_steps] = action_log_probs
				mean_entropy += dist.entropy().sum()
			action_log_probs = combined_log_probs
			dist_entropy = mean_entropy/B
		else:
			dist_entropy = dist.entropy().mean()

		return value, action_log_probs, dist_entropy, rnn_hxs
