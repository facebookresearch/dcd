# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import gym
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from .obs_wrappers import AdversarialObservationWrapper


class MultiGridFullyObsWrapper(AdversarialObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env, is_adversarial=True):
        super().__init__(env)

        self.is_adversarial = is_adversarial

        self.observation_space.spaces["full_obs"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

    def agent_observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()

        # Note env.agent_pos is an array of length K, for K multigrid agents
        if env.agent_pos[0] is not None:
            full_grid[env.agent_pos[0][0]][env.agent_pos[0][1]] = np.array(
                [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir[0]]
            )

        obs["full_obs"] = full_grid

        return obs

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.is_adversarial:
            return observation
        else:
            return self.agent_observation(observation)
