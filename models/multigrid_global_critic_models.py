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


class MultigridGlobalCriticNetwork(DeviceAwareModule):
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
        use_global_policy=False,
    ):
        super(MultigridGlobalCriticNetwork, self).__init__()

        num_actions = action_space.n

        self.use_global_policy = use_global_policy

        # Image embedding
        obs_shape = observation_space["image"].shape
        m = obs_shape[-2]  # x input dim
        n = obs_shape[-1]  # y input dim
        c = obs_shape[-3]  # channel input dim

        # Full obs embedding
        full_obs_shape = observation_space["full_obs"].shape
        global_m = full_obs_shape[-2]
        global_n = full_obs_shape[-1]
        global_c = full_obs_shape[-3]

        self.global_image_conv = nn.Sequential(
            Conv2d_tf(3, 8, kernel_size=2, stride=2, padding="VALID"),
            nn.ReLU(),
            Conv2d_tf(8, 16, kernel_size=3, stride=1, padding="VALID"),
            nn.Flatten(),
        )
        self.global_image_embedding_size = (
            (((((global_n - 2) // 2) + 1) - 3) + 1)
            * (((((global_n - 2) // 2) + 1) - 3) + 1)
            * 16
        )

        if self.use_global_policy:
            self.image_conv = self.global_image_conv
            self.image_embedding_size = self.global_image_embedding_size
            self.preprocessed_input_size = self.image_embedding_size
        else:
            self.image_conv = nn.Sequential(
                Conv2d_tf(
                    3,
                    conv_filters,
                    kernel_size=conv_kernel_size,
                    stride=1,
                    padding="VALID",
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

        # RNN (only for policy)
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
        if self.use_global_policy:
            self.global_base_output_size = self.base_output_size
        else:
            self.global_base_output_size = (
                self.global_image_embedding_size + self.base_output_size
            )
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(
                value_fc_layers, input_size=self.global_base_output_size
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
        if self.use_global_policy:
            image = inputs.get("full_obs", None)
        else:
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

        global_image = inputs.get("full_obs", None)
        if global_image is not None:
            if self.use_global_policy:
                global_core_features = core_features
            else:
                in_global_image = self.global_image_conv(global_image)
                global_core_features = torch.cat(
                    (core_features, in_global_image), dim=-1
                )
        else:
            global_core_features = None

        return core_features, rnn_hxs, global_core_features

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        core_features, rnn_hxs, global_core_features = self._forward_base(
            inputs, rnn_hxs, masks
        )

        dist = self.actor(core_features)
        if global_core_features is not None:
            value = self.critic(global_core_features)
        else:
            value = 0

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_dist = dist.logits

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs, global_core_features = self._forward_base(
            inputs, rnn_hxs, masks
        )

        if global_core_features is not None:
            value = self.critic(global_core_features)
        else:
            value = 0

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs, global_core_features = self._forward_base(
            inputs, rnn_hxs, masks
        )

        dist = self.actor(core_features)
        if global_core_features is not None:
            value = self.critic(global_core_features)
        else:
            value = 0

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
