# Copyright (c) 2020 Tianshou contributors
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of
# https://github.com/marlbenchmark/on-policy/blob/0fc8a9355bb7ce2589716eeb543f498edcc91dc6/onpolicy/algorithms/utils/popart.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DeviceAwareModule


class PopArt(DeviceAwareModule):

    def __init__(
        self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5
    ):

        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))
        self.bias = nn.Parameter(torch.Tensor(output_shape))

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(self.device)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(self.device)

        old_mean, old_stddev = self.mean, self.stddev

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev.data = (self.mean_sq - self.mean**2).sqrt().clamp(min=1e-4)

        self.weight.data = self.weight * old_stddev / self.stddev
        self.bias.data = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(self.device)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)

        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(self.device)

        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var) + mean

        return out
