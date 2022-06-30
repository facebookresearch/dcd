# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version code found in
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from gym.spaces import MultiDiscrete
from collections import namedtuple

from .common import *
from torch.distributions.normal import Normal
from .popart import PopArt

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(DeviceAwareModule):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

## Policy from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BipedalWalkerStudentPolicy(DeviceAwareModule):
    def __init__(self, obs_shape, action_space, recurrent=False,base_kwargs=None):
        super(BipedalWalkerStudentPolicy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(obs_shape[0], recurrent=recurrent, **base_kwargs)

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)

        if recurrent:
            RNN = namedtuple("RNN", 'arch')
            self.rnn = RNN('gru')

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs):
        value, action, action_log_probs, rnn_hxs = self.act(inputs, rnn_hxs=None, masks=None, deterministic=False)
        return action

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class BipedalWalkerAdversaryPolicy(DeviceAwareModule):
    def __init__(self, observation_space, action_space, editor=False, random=False, base_kwargs=None):
        super(BipedalWalkerAdversaryPolicy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        self.random = random

        self.design_dim = observation_space['image'].shape[0]
        self.random_z_dim = observation_space['random_z'].shape[0]

        obs_dim = self.design_dim + self.random_z_dim  + 1

        self.base = MLPBase(obs_dim, **base_kwargs)

        self.editor = editor

        self.action_dim = action_space.shape[0]
        if self.editor:
            self.dist = [Categorical(self.base.output_size, x) for x in action_space.nvec]
            self.action_space = action_space
        else:
            self.dist = DiagGaussian(self.base.output_size, self.action_dim)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def preprocess(self, inputs):
        obs = torch.cat([inputs['image'], inputs['random_z'], inputs['time_step']], axis=1)
        return obs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        inputs = self.preprocess(inputs)

        if self.random:
            if self.editor:
                B = inputs.shape[0]
                action = torch.zeros((B, 2), dtype=torch.int64, device=self.device)
                action_log_dist = torch.ones(B, self.action_space.nvec[0] + self.action_space.nvec[1], device=self.device)
                for b in range(B):
                    action[b] = torch.tensor(self.action_space.sample()).to(self.device)
            else:
                action = torch.tensor(np.random.uniform(-1,1, inputs.shape[0]), device=self.device).reshape(-1,1)
                action_log_dist = torch.ones(inputs.shape[0], self.action_dim, device=self.device)
            values = torch.zeros(inputs.shape[0], 1, device=self.device)
            return values, action, action_log_dist, rnn_hxs

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        if self.editor:
            dist = [fwd(actor_features) for fwd in self.dist]
        else:
            dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action = F.tanh(action)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        inputs = self.preprocess(inputs)
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        inputs = self.preprocess(inputs)
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.rnn = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.rnn(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
