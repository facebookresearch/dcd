# Copyright (c) 2019 Antonin Raffin
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is an extended version of
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_frame_stack.py

from .vec_env import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, obs_key=None):
        self.venv = venv
        self.n_frame_channels = venv.observation_space.shape[-1]
        self.nstack = nstack
        self.obs_key = obs_key
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        if self.obs_key:
            obs = obs[obs_key]
        self.stackedobs = np.roll(self.stackedobs, shift=-self.n_frame_channels, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self, seed=None, index=None):
        if seed is not None and index is not None:
            obs = self.venv.seed(seed, index)
            if self.obs_key:
                obs = obs[obs_key]
            self.stackedobs[index] = 0
            self.stackedobs[index,...,-obs.shape[-1]:] = obs
            return self.stackedobs[index,:]
        else:
            obs = self.venv.reset()
            if self.obs_key:
                obs = obs[obs_key]
            self.stackedobs[...] = 0
            self.stackedobs[..., -obs.shape[-1]:] = obs
            return self.stackedobs

    def reset_agent(self):
        obs = self.venv.reset_agent()
        if self.obs_key:
            obs = obs[obs_key]
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def reset_random(self):
        obs = self.venv.reset_random()
        if self.obs_key:
            obs = obs[obs_key]
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
