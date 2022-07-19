# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a heavily modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py


from collections import defaultdict

import numpy as np
import torch
import gym

from torch.utils.data.sampler import \
    BatchSampler, SubsetRandomSampler, SequentialSampler
from lempel_ziv_complexity import lempel_ziv_complexity 

def to_tensor(a):
    if isinstance(a, dict):
        for k in a.keys():
            if isinstance(a[k], np.ndarray):
                a[k] = torch.from_numpy(a[k]).float()
    elif isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    elif isinstance(a, list):
        a = torch.tensor(a, dtype=torch.float)

    return a

def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict): 
        return {k: _tensor[k].view(T * N, *_tensor[k].size()[2:]) for k in _tensor.keys()}
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage(object):
    def __init__(self, 
                 model,
                 num_steps, num_processes, observation_space, action_space, 
                 recurrent_hidden_state_size, recurrent_arch='rnn', 
                 use_proper_time_limits=False, 
                 use_popart=False,
                 device='cpu'):

        self.device = device
        self.model = model
        self.num_processes = num_processes
        self.recurrent_arch = recurrent_arch
        self.recurrent_hidden_state_size = recurrent_hidden_state_size
        self.is_lstm = recurrent_arch == 'lstm'
        recurrent_hidden_state_buffer_size = 2*recurrent_hidden_state_size if self.is_lstm \
            else recurrent_hidden_state_size
        self.use_proper_time_limits = use_proper_time_limits
        self.use_popart = use_popart

        self.truncated_obs = None
        if isinstance(observation_space, dict):
            self.is_dict_obs = True
            self.obs = {k:torch.zeros(num_steps + 1, num_processes, *(observation_space[k]).shape) \
                for k,obs in observation_space.items()}

            if self.use_proper_time_limits:
                self.truncated_obs = {k:torch.zeros(num_steps + 1, num_processes, *(observation_space[k]).shape) \
                    for k,obs in observation_space.items()}
        else:
            self.is_dict_obs = False
            self.obs = torch.zeros(num_steps + 1, num_processes, *observation_space.shape)

            if self.use_proper_time_limits:
                self.truncated_obs = torch.zeros_like(self.obs)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_buffer_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
            self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)
        else: # Hack it to just store action prob for sampled action if continuous
            action_shape = action_space.shape[0]
            self.action_log_dist = torch.zeros(num_steps, num_processes, 1)

        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        # Keep track of cliffhanger timesteps
        self.cliffhanger_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.truncated_value_preds = None
        if self.use_proper_time_limits:
            self.truncated_value_preds = torch.zeros_like(self.value_preds)

        self.denorm_value_preds = None

        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.device = device

        if self.is_dict_obs:
            for k, obs in self.obs.items():
                self.obs[k] = obs.to(device)
        else:
            self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.cliffhanger_masks = self.cliffhanger_masks.to(device)
        self.level_seeds = self.level_seeds.to(device)

        if self.use_proper_time_limits:
            if self.is_dict_obs:
                for k, obs in self.truncated_obs.items():
                    self.truncated_obs[k] = obs.to(device)
            else:
                self.truncated_obs = self.truncated_obs.to(device)

            self.truncated_value_preds = self.truncated_value_preds.to(device)

    def get_obs(self, idx):
        if self.is_dict_obs:
            return {k: self.obs[k][idx] for k in self.obs.keys()}
        else:
            return self.obs[idx]

    def copy_obs_to_index(self, obs, index):
        if self.is_dict_obs:
            [self.obs[k][index].copy_(obs[k]) for k in self.obs.keys()]
        else:
            self.obs[index].copy_(obs)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, action_log_dist,
               value_preds, rewards, masks, bad_masks, level_seeds=None, cliffhanger_masks=None):
        if len(rewards.shape) == 3: rewards = rewards.squeeze(2)

        if self.is_dict_obs:
            [self.obs[k][self.step + 1].copy_(obs[k]) for k in self.obs.keys()]
        else:
            self.obs[self.step + 1].copy_(obs)

        if self.is_lstm:
            self.recurrent_hidden_states[self.step +1,:,
                :self.recurrent_hidden_state_size].copy_(recurrent_hidden_states[0])
            self.recurrent_hidden_states[self.step +1,:,
                self.recurrent_hidden_state_size:].copy_(recurrent_hidden_states[1])
        else:
            self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)

        self.actions[self.step].copy_(actions) 
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.action_log_dist[self.step].copy_(action_log_dist)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        if cliffhanger_masks is not None:
            self.cliffhanger_masks[self.step + 1].copy_(cliffhanger_masks)

        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def insert_truncated_obs(self, obs, index):
        if self.is_dict_obs:
            [self.truncated_obs[k][self.step + 1][index].copy_(
                to_tensor(obs[k])) for k in self.truncated_obs.keys()]
        else:
            self.truncated_obs[self.step + 1][index].copy_(to_tensor(obs))

    def after_update(self):
        if self.is_dict_obs:
            [self.obs[k][0].copy_(self.obs[k][-1]) for k in self.obs.keys()]
        else:
            self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.cliffhanger_masks[0].copy_(self.cliffhanger_masks[-1])

    def replace_final_return(self, returns):
        self.rewards[-1] = returns

    def _compute_truncated_value_preds(self):
        self.truncated_value_preds.copy_(self.value_preds)
        with torch.no_grad():
            # For each process, forward truncated obs
            for i in range(self.num_processes):
                steps = (self.bad_masks[:,i,0] == 0).nonzero().squeeze()
                if len(steps.shape) == 0 or steps.shape[0] == 0:
                    continue

                if self.is_dict_obs:
                    obs = {k:self.truncated_obs[k][steps.squeeze(), i, :] 
                        for k in self.truncated_obs.keys()}
                else:
                    obs = self.truncated_obs[steps.squeeze(),i,:]

                rnn_hxs = self.recurrent_hidden_states[steps,i,:]
                if self.is_lstm:
                    rnn_hxs = self._split_batched_lstm_recurrent_hidden_states(rnn_hxs)
                masks = torch.ones((len(steps), 1), device=self.device)
                value_preds = self.model.get_value(obs, rnn_hxs, masks)

                self.truncated_value_preds[steps,i,:] = value_preds

        return self.truncated_value_preds

    def compute_gae_returns(self, 
                            returns_buffer,
                            next_value, 
                            gamma, 
                            gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        value_preds = self.value_preds

        if self.use_proper_time_limits:
            # Get truncated value preds
            self._compute_truncated_value_preds()
            value_preds = self.truncated_value_preds

        if self.use_popart:
            self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions
            value_preds = self.denorm_value_preds

        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + \
                gamma*value_preds[step + 1]*self.masks[step + 1] - value_preds[step]

            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_preds[step]

    def compute_discounted_returns(self,
                                   returns_buffer, 
                                   next_value,
                                   gamma):
        self.value_preds[-1] = next_value
        value_preds = self.value_preds

        if self.use_proper_time_limits:    
            self._compute_truncated_value_preds()
            value_preds = self.truncated_value_preds

        if self.use_popart:
            self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions

        self.returns[-1] = value_preds[-1]

        for step in reversed(range(self.rewards.size(0))):
            returns_buffer[step] = returns_buffer[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step]

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda):
        if use_gae:
            self.compute_gae_returns(
                self.returns, next_value, gamma, gae_lambda)
        else:
            self.compute_discounted_returns(
                self.returns, next_value, gamma)

    def get_batched_value_loss(self, 
            signed=False, 
            positive_only=False, 
            power=1, 
            clipped=True, 
            batched=True):
        """
        Assumes buffer contains pre-computed returns via compute_returns.
        Computes the mean episodic value loss per batch.
        """

        # If agent uses popart, then value_preds are normalized, while 
        # returns are not.
        if self.use_popart:
            value_preds = self.denorm_value_preds[:-1]
        else:
            value_preds = self.value_preds[:-1]

        returns = self.returns[:-1]

        if signed:
            td = returns - value_preds
        elif positive_only:
            td = (returns - value_preds).clamp(0)
        else:
            td = (returns - value_preds).abs()
        if power > 1:
            td = td**power

        batch_td = td.mean(0) # B x 1

        if clipped:
            batch_td = torch.clamp(batch_td, -1, 1) 
        
        if batched:
            return batch_td
        else:
            return batch_td.mean().item()

    def get_batched_action_complexity(self):
        """
        Returns per-batch LZ complexity scores of the action trajectories in the buffer
        """
        num_processes = self.actions.shape[1]
        batched_complexity = torch.zeros(num_processes, 1, dtype=torch.float)
        for b in range(num_processes):
            num_traj = 0
            avg_complexity = 0
            done_steps = [0] + (self.masks[:,b,0] == 0).nonzero().flatten().tolist()
            for i, t in enumerate(done_steps[:-1]):
                if len(done_steps) > 1:
                    next_done = done_steps[i+1]
                else:
                    next_done = self.actions.shape[0]
                action_str = ' '.join([str(a.item()) for a in self.actions[t:next_done,b,0]])
                avg_complexity += lempel_ziv_complexity(action_str)
                num_traj += 1
            batched_complexity[b] = avg_complexity/num_traj

        return batched_complexity

    def get_action_complexity(self):
        """
        Returns mean LZ complexity scores of the action trajectories in the buffer
        """
        num_processes = self.actions.shape[1]
        avg_complexity = 0
        num_traj = 0
        for b in range(num_processes):
            done_steps = [0] + (self.masks[:,b,0] == 0).nonzero().flatten().tolist()
            for i, t in enumerate(done_steps[:-1]):
                if len(done_steps) > 1:
                    next_done = done_steps[i+1]
                else:
                    next_done = self.actions.shape[0]
                action_str = ' '.join([str(a.item()) for a in self.actions[t:next_done,b,0]])
                avg_complexity += lempel_ziv_complexity(action_str)
                num_traj += 1
    
        return avg_complexity/num_traj

    def get_action_traj(self, as_string=False):
        if as_string:
            num_processes = self.actions.shape[1]
            traj = []
            for b in range(num_processes):
                action_str = ' '.join([str(a.item()) for a in self.actions[:,b,0]])
                traj.append(action_str)
            return traj
        else:
            return self.actions.squeeze(-1)

    def _split_batched_lstm_recurrent_hidden_states(self, hxs):
        return (hxs[:, :self.recurrent_hidden_state_size],
                hxs[:, self.recurrent_hidden_state_size:])

    def get_recurrent_hidden_state(self, step):
        if self.is_lstm:
            return self._split_batched_lstm_recurrent_hidden_states(
                    self.recurrent_hidden_states[step,:].squeeze(0))
        return self.recurrent_hidden_states[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
     
        for indices in sampler:
            if self.is_dict_obs:
                obs_batch = {k: self.obs[k][:-1].view(-1, *self.obs[k].size()[2:])[indices] for k in self.obs.keys()}
            else:
                obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]

            actions_batch = self.actions.view(-1,
                                            self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]

            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            if self.is_lstm: 
                # Split into (hxs, cxs) for LSTM
                recurrent_hidden_states_batch = \
                    self._split_batched_lstm_recurrent_hidden_states(recurrent_hidden_states_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            if self.is_dict_obs:
                obs_batch = defaultdict(list)
            else:
                obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self.is_dict_obs:
                    [obs_batch[k].append(self.obs[k][:-1,ind]) for k in self.obs.keys()]
                else:
                    obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            if self.is_dict_obs:
                for k in obs_batch.keys():
                    obs_batch[k] = torch.stack(obs_batch[k],1)
            else:
                obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            if self.is_lstm: 
                # Split into (hxs, cxs) for LSTM
                recurrent_hidden_states_batch = \
                    self._split_batched_lstm_recurrent_hidden_states(recurrent_hidden_states_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
