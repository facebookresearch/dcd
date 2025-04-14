# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
from envs.registration import register as gym_register

env_list = []


def register(env_id, entry_point, reward_threshold=0.95, max_episode_steps=None):
    assert env_id.startswith("MultiGrid-")
    if env_id in env_list:
        del gym.envs.registry.env_specs[id]
    else:
        env_list.append(id)

    kwargs = dict(id=env_id, entry_point=entry_point, reward_threshold=reward_threshold)

    if max_episode_steps:
        kwargs.update({"max_episode_steps": max_episode_steps})

    gym_register(**kwargs)
