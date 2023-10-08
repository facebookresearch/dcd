# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 

class PPO():
    """
    Vanilla PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 kl_loss_coef=0.0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_value_loss=True,
                 log_grad_norm=False):

        self.actor_critic = actor_critic    

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_value_loss = clip_value_loss

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_loss_coef = kl_loss_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.log_grad_norm = log_grad_norm

    def _grad_norm(self):
        total_norm = 0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def update(self, rollouts, discard_grad=False, kl_dict=None):
        use_kl_loss = (kl_dict is not None) and (self.kl_loss_coef > 0.0)
        
        if rollouts.use_popart:
            value_preds = rollouts.denorm_value_preds
        else:
            value_preds = rollouts.value_preds

        advantages = rollouts.returns[:-1] - value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        if use_kl_loss:
            kl_loss_epoch = 0

        if self.log_grad_norm:
            grad_norms = []

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
                if use_kl_loss:
                    values, action_log_probs, dist_entropy, _, dist_protagonist = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, return_policy_logits=True)
                    with torch.no_grad():
                        _, _, _, _, dist_antagonist = kl_dict['antagonist_model'].evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, return_policy_logits=True)
                else:
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if rollouts.use_popart:
                    self.actor_critic.popart.update(return_batch)
                    return_batch = self.actor_critic.popart.normalize(return_batch)

                if self.clip_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                else:
                    value_loss = F.smooth_l1_loss(values, return_batch)

                if use_kl_loss:
                    kl_div = torch.distributions.kl.kl_divergence(dist_antagonist, dist_protagonist)
                    bs = kl_div.shape[0]
                    kl_loss = kl_div.sum() / bs
                self.optimizer.zero_grad()
                loss = (value_loss*self.value_loss_coef + action_loss - dist_entropy*self.entropy_coef)
                if use_kl_loss:
                    loss += (self.kl_loss_coef*kl_loss)

                loss.backward()

                if self.log_grad_norm:
                    grad_norms.append(self._grad_norm())

                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                    
                if not discard_grad:
                    self.optimizer.step()
                                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if use_kl_loss:
                    kl_loss_epoch += kl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        if use_kl_loss:
            kl_loss_epoch /= num_updates

        info = {}
        if self.log_grad_norm:
            info = {'grad_norms': grad_norms}
        if use_kl_loss:
            info['kl_loss'] = kl_loss_epoch

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, info
