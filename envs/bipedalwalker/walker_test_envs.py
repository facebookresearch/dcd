# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
import time
import numpy as np
import torch

from .walker_env import EnvConfig, BipedalWalkerCustom
from envs.registration import register as gym_register


def get_config(
    name="default",
    ground_roughness=0,
    pit_gap=[],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[],
    stair_width=[],
    stair_steps=[],
):

    config = EnvConfig(
        name=name,
        ground_roughness=ground_roughness,
        pit_gap=pit_gap,
        stump_width=stump_width,
        stump_height=stump_height,
        stump_float=stump_float,
        stair_height=stair_height,
        stair_width=stair_width,
        stair_steps=stair_steps,
    )

    return config


class BipedalWalkerDefault(BipedalWalkerCustom):
    def __init__(self):
        config = get_config()
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


## stump height
class BipedalWalkerMedStumps(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(stump_height=[2, 2], stump_width=[1, 2], stump_float=[0, 1])
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


class BipedalWalkerHighStumps(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(stump_height=[5, 5], stump_width=[1, 2], stump_float=[0, 1])
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


## pit gap
class BipedalWalkerMedPits(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(pit_gap=[5, 5])
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


class BipedalWalkerWidePits(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(pit_gap=[10, 10])
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


# stair height + number of stairs
class BipedalWalkerMedStairs(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(stair_height=[2, 2], stair_steps=[5], stair_width=[4, 5])
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


class BipedalWalkerHighStairs(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(stair_height=[5, 5], stair_steps=[9], stair_width=[4, 5])
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


# ground roughness
class BipedalWalkerMedRoughness(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(ground_roughness=5)
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


class BipedalWalkerHighRoughness(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(ground_roughness=9)
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


# everything maxed out
class BipedalWalkerInsane(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(
            stump_height=[5, 5],
            stump_width=[1, 2],
            stump_float=[0, 1],
            pit_gap=[10, 10],
            stair_height=[5, 5],
            stair_steps=[9],
            stair_width=[4, 5],
            ground_roughness=9,
        )
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


## PCG "Extremely Challenging" Env
# First samples params, then generates level
class BipedalWalkerXChal(BipedalWalkerCustom):
    def __init__(self):
        config = get_config(
            stump_height=[],
            stump_width=[],
            stump_float=[],
            pit_gap=[],
            stair_height=[],
            stair_steps=0,
            stair_width=[],
            ground_roughness=0,
        )
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        self.level_seed = int(str(time.time() / 1000)[-3:])
        super().seed(self.level_seed)

        stump_high = np.random.uniform(2.4, 3)
        gap_high = np.random.uniform(6, 8)
        roughness = np.random.uniform(4.5, 8)

        config = get_config(
            stump_height=[0, stump_high],
            stump_width=[1, 2],
            stump_float=[0, 1],
            pit_gap=[0, gap_high],
            stair_height=[],
            stair_steps=0,
            stair_width=[],
            ground_roughness=roughness,
        )

        super().re_init(config, self.level_seed)
        return super()._reset_env()


## POET Rose
roses = {
    "1a": [5.6, 2.4, 2.82, 6.4, 4.48],
    "1b": [5.44, 1.8, 2.82, 6.72, 4.48],
    "2a": [7.2, 1.98, 2.82, 7.2, 5.6],
    "2b": [5.76, 2.16, 2.76, 7.2, 1.6],
    "3a": [5.28, 1.98, 2.76, 7.2, 4.8],
    "3b": [4.8, 2.4, 2.76, 4.48, 4.8],
}


class BipedalWalkerPOETRose(BipedalWalkerCustom):
    def __init__(self, rose_id="1a"):
        id = roses[rose_id]
        config = get_config(
            stump_height=[id[1], id[2]],
            stump_width=[1, 2],
            stump_float=[0, 1],
            pit_gap=[id[4], id[3]],
            stair_height=[],
            stair_steps=[],
            stair_width=[],
            ground_roughness=id[0],
        )
        super().__init__(env_config=config, seed=int(str(time.time() / 1000)[-3:]))

    def reset(self):
        super().seed(int(str(time.time() / 1000)[-3:]))
        return super()._reset_env()


class BipedalWalkerPOETRose1a(BipedalWalkerPOETRose):
    def __init__(self):
        super().__init__(rose_id="1a")


class BipedalWalkerPOETRose1b(BipedalWalkerPOETRose):
    def __init__(self):
        super().__init__(rose_id="1b")


class BipedalWalkerPOETRose2a(BipedalWalkerPOETRose):
    def __init__(self):
        super().__init__(rose_id="2a")


class BipedalWalkerPOETRose2b(BipedalWalkerPOETRose):
    def __init__(self):
        super().__init__(rose_id="2b")


class BipedalWalkerPOETRose3a(BipedalWalkerPOETRose):
    def __init__(self):
        super().__init__(rose_id="3a")


class BipedalWalkerPOETRose3b(BipedalWalkerPOETRose):
    def __init__(self):
        super().__init__(rose_id="3b")


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

gym_register(
    id="BipedalWalker-Default-v0",
    entry_point=module_path + ":BipedalWalkerDefault",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-Med-Roughness-v0",
    entry_point=module_path + ":BipedalWalkerMedRoughness",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-High-Roughness-v0",
    entry_point=module_path + ":BipedalWalkerHighRoughness",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-Med-StumpHeight-v0",
    entry_point=module_path + ":BipedalWalkerMedStumps",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-High-StumpHeight-v0",
    entry_point=module_path + ":BipedalWalkerHighStumps",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-Med-Stairs-v0",
    entry_point=module_path + ":BipedalWalkerMedStairs",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-High-Stairs-v0",
    entry_point=module_path + ":BipedalWalkerHighStairs",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-Med-PitGap-v0",
    entry_point=module_path + ":BipedalWalkerMedPits",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-Wide-PitGap-v0",
    entry_point=module_path + ":BipedalWalkerWidePits",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-Insane-v0",
    entry_point=module_path + ":BipedalWalkerInsane",
    max_episode_steps=2000,
)

gym_register(
    id="BipedalWalker-XChal-v0",
    entry_point=module_path + ":BipedalWalkerXChal",
    max_episode_steps=2000,
)

for id in ["1a", "1b", "2a", "2b", "3a", "3b"]:
    gym_register(
        id=f"BipedalWalker-POET-Rose-{id}-v0",
        entry_point=module_path + f":BipedalWalkerPOETRose{id}",
        max_episode_steps=2000,
    )
