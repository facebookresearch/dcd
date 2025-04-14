# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import gym

from .vec_env import VecEnvWrapper


class AdversarialObservationWrapper(gym.core.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.agent_observation(observation), reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def reset_agent(self, **kwargs):
        observation = self.env.reset_agent(**kwargs)
        return self.agent_observation(observation)

    def reset_random(self, **kwargs):
        observation = self.env.reset_random(**kwargs)
        return self.agent_observation(observation)

    def reset_to_level(self, level, **kwargs):
        observation = self.env.reset_to_level(level, **kwargs)
        return self.agent_observation(observation)

    def agent_observation(self, observation):
        raise NotImplementedError


class VecPreprocessImageWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv,
        obs_key=None,
        transpose_order=None,
        scale=None,
        channel_first=False,
        to_tensor=True,
        device=None,
    ):
        super().__init__(venv)

        self.is_dict_obs = isinstance(venv.observation_space, gym.spaces.Dict)

        self.transpose_order = transpose_order
        if self.transpose_order:
            self.batch_transpose_order = [
                0,
            ] + list([i + 1 for i in transpose_order])
        else:
            self.batch_transpose_order = None
        self.obs_key = obs_key
        self._obs_space = None
        self._adversary_obs_space = None
        self.to_tensor = to_tensor
        self.device = device

        # Colorspace parameters
        self.scale = scale
        self.channel_first = channel_first
        self.channel_index = 1 if channel_first else -1

        image_obs_space = self.venv.observation_space
        if self.obs_key:
            image_obs_space = image_obs_space[self.obs_key]
        self.num_channels = image_obs_space.shape[self.channel_index]

        delattr(self, "observation_space")

    def _obs_dict_to_tensor(self, obs):
        for k in obs.keys():
            if isinstance(obs[k], np.ndarray):
                obs[k] = torch.from_numpy(obs[k]).float()
                if self.device:
                    obs[k] = obs[k].to(self.device)
        return obs

    def _transpose(self, obs):
        if len(obs.shape) == len(self.batch_transpose_order):
            return obs.transpose(*self.batch_transpose_order)
        else:
            return obs.transpose(*self.transpose_order)

    def _preprocess(self, obs, obs_key=None):
        if obs_key is None:
            if self.scale:
                obs = obs / self.scale

            if self.batch_transpose_order:
                # obs = obs.transpose(*self.batch_transpose_order)
                obs = self._transpose(obs)

            if isinstance(obs, np.ndarray) and self.to_tensor:
                obs = torch.from_numpy(obs).float()
                if self.device:
                    obs = obs.to(self.device)
            elif isinstance(obs, dict) and self.to_tensor:
                obs = self._obs_dict_to_tensor(obs)
        else:
            if self.scale:
                obs[self.obs_key] = obs[self.obs_key] / self.scale

            if self.batch_transpose_order:
                obs[self.obs_key] = self._transpose(obs[self.obs_key])
                if "full_obs" in obs:
                    obs["full_obs"] = self._transpose(obs["full_obs"])

            if self.to_tensor:
                obs = self._obs_dict_to_tensor(obs)

        return obs

    def _transpose_box_space(self, space):
        if isinstance(space, gym.spaces.Box):
            shape = np.array(space.shape)
            shape = shape[self.transpose_order]
            return gym.spaces.Box(low=0, high=255, shape=shape, dtype="uint8")
        else:
            raise ValueError("Expected gym.spaces.Box")

    def _transpose_obs_space(self, obs_space):
        if self.obs_key:
            if isinstance(obs_space, gym.spaces.Dict):
                keys = obs_space.spaces
            else:
                keys = obs_space.keys()
            transposed_obs_space = {k: obs_space[k] for k in keys}
            transposed_obs_space[self.obs_key] = self._transpose_box_space(
                transposed_obs_space[self.obs_key]
            )

            if "full_obs" in transposed_obs_space:
                transposed_obs_space["full_obs"] = self._transpose_box_space(
                    transposed_obs_space["full_obs"]
                )
        else:
            transposed_obs_space = self._transpose_box_space(obs_space)

        return transposed_obs_space

    # Public interface
    def reset(self):
        obs = self.venv.reset()
        return self._preprocess(obs, obs_key=self.obs_key)

    def reset_random(self):
        obs = self.venv.reset_random()
        return self._preprocess(obs, obs_key=self.obs_key)

    def reset_agent(self):
        obs = self.venv.reset_agent()
        return self._preprocess(obs, obs_key=self.obs_key)

    def reset_to_level(self, level, index):
        obs = self.venv.reset_to_level(level, index)
        return self._preprocess(obs, obs_key=self.obs_key)

    def reset_to_level_batch(self, level):
        obs = self.venv.reset_to_level_batch(level)
        return self._preprocess(obs, obs_key=self.obs_key)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = self._preprocess(obs, obs_key=self.obs_key)

        for i, info in enumerate(infos):
            if "truncated_obs" in info:
                truncated_obs = info["truncated_obs"]
                infos[i]["truncated_obs"] = self._preprocess(
                    truncated_obs, obs_key=self.obs_key
                )

        if self.to_tensor:
            rews = torch.from_numpy(rews).unsqueeze(dim=1).float()

        return obs, rews, dones, infos

    def step_adversary(self, action):
        obs, rews, dones, infos = self.venv.step_adversary(action)
        obs = self._preprocess(obs, obs_key=self.obs_key)

        if self.to_tensor:
            rews = torch.from_numpy(rews).unsqueeze(dim=1).float()

        return obs, rews, dones, infos

    def get_observation_space(self):
        if self._obs_space:
            return self._obs_space

        obs_space = self.venv.observation_space

        if self.batch_transpose_order:
            self._obs_space = self._transpose_obs_space(obs_space)
        else:
            self._obs_space = obs_space

        return self._obs_space

    def get_adversary_observation_space(self):
        if self._adversary_obs_space:
            return self._adversary_obs_space

        adversary_obs_space = self.venv.adversary_observation_space
        obs_space = self.venv.observation_space
        same_shape = (
            hasattr(adversary_obs_space, "shape")
            and hasattr(obs_space, "shape")
            and adversary_obs_space.shape == obs_space.shape
        )
        same_obs_key = self.obs_key and self.obs_key in adversary_obs_space

        if self.batch_transpose_order and (same_shape or same_obs_key):
            self._adversary_obs_space = self._transpose_obs_space(adversary_obs_space)
        else:
            self._adversary_obs_space = adversary_obs_space

        return self._adversary_obs_space

    def __getattr__(self, name):
        if name == "observation_space":
            return self.get_observation_space()
        elif name == "adversary_observation_space":
            return self.get_adversary_observation_space()
        elif name == "adversary_action_space":
            return self.venv.get_adversary_action_space()
        else:
            return getattr(self.venv, name)
