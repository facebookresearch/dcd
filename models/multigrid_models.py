# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Categorical
from .common import *


class MultigridNetwork(DeviceAwareModule):
    """
    Actor-Critic module
    """

    def __init__(
        self,
        observation_space,
        action_space,
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),
        conv_filters=16,
        conv_kernel_size=3,
        scalar_fc=5,
        scalar_dim=4,
        random_z_dim=0,
        xy_dim=0,
        recurrent_arch="lstm",
        recurrent_hidden_size=256,
        random=False,
    ):
        super(MultigridNetwork, self).__init__()

        self.random = random
        self.action_space = action_space
        num_actions = action_space.n

        # Image embeddings
        obs_shape = observation_space["image"].shape
        m = obs_shape[-2]  # x input dim
        n = obs_shape[-1]  # y input dim
        c = obs_shape[-3]  # channel input dim

        self.image_conv = nn.Sequential(
            Conv2d_tf(
                3, conv_filters, kernel_size=conv_kernel_size, stride=1, padding="valid"
            ),
            nn.Flatten(),
            nn.ReLU(),
        )
        self.image_embedding_size = (
            (n - conv_kernel_size + 1) * (m - conv_kernel_size + 1) * conv_filters
        )
        self.preprocessed_input_size = self.image_embedding_size

        # x, y positional embeddings
        self.xy_embed = None
        self.xy_dim = xy_dim
        if xy_dim:
            self.preprocessed_input_size += 2 * xy_dim

        # Scalar embedding
        self.scalar_embed = None
        self.scalar_dim = scalar_dim
        if scalar_dim:
            self.scalar_embed = nn.Linear(scalar_dim, scalar_fc)
            self.preprocessed_input_size += scalar_fc

        self.preprocessed_input_size += random_z_dim
        self.base_output_size = self.preprocessed_input_size

        # RNN
        self.rnn = None
        if recurrent_arch:
            self.rnn = RNN(
                input_size=self.preprocessed_input_size,
                hidden_size=recurrent_hidden_size,
                arch=recurrent_arch,
            )
            self.base_output_size = recurrent_hidden_size

        # Policy head
        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(
                actor_fc_layers, input_size=self.base_output_size
            ),
            Categorical(actor_fc_layers[-1], num_actions),
        )

        # Value head
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(
                value_fc_layers, input_size=self.base_output_size
            ),
            init_(nn.Linear(value_fc_layers[-1], 1)),
        )

        apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        # """Size of rnn_hx."""
        if self.rnn is not None:
            return self.rnn.recurrent_hidden_state_size
        else:
            return 0

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _forward_base(self, inputs, rnn_hxs, masks):
        # Unpack input key values
        image = inputs.get("image")

        scalar = inputs.get("direction")
        if scalar is None:
            scalar = inputs.get("time_step")

        x = inputs.get("x")
        y = inputs.get("y")

        in_z = inputs.get("random_z", torch.tensor([], device=self.device))

        in_image = self.image_conv(image)

        if self.xy_embed:
            x = one_hot(self.xy_dim, x, device=self.device)
            y = one_hot(self.xy_dim, y, device=self.device)
            in_x = self.xy_embed(x)
            in_y = self.xy_embed(y)
        else:
            in_x = torch.tensor([], device=self.device)
            in_y = torch.tensor([], device=self.device)

        if self.scalar_embed:
            in_scalar = one_hot(self.scalar_dim, scalar).to(self.device)
            in_scalar = self.scalar_embed(in_scalar)
        else:
            in_scalar = torch.tensor([], device=self.device)

        in_embedded = torch.cat((in_image, in_x, in_y, in_scalar, in_z), dim=-1)

        if self.rnn is not None:
            core_features, rnn_hxs = self.rnn(in_embedded, rnn_hxs, masks)
        else:
            core_features = in_embedded

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            B = inputs["image"].shape[0]
            action = torch.zeros((B, 1), dtype=torch.int64, device=self.device)
            values = torch.zeros((B, 1), device=self.device)
            action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
            for b in range(B):
                action[b] = self.action_space.sample()

            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_dist = dist.logits

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(
        self, inputs, rnn_hxs, masks, action, return_policy_logits=False
    ):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist

        return value, action_log_probs, dist_entropy, rnn_hxs
