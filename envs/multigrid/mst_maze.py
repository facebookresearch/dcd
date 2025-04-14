# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import networkx
import gym_minigrid.minigrid as minigrid

from . import multigrid
from util.unionfind import UnionFind
import envs.registration as register


class MSTMazeEnv(multigrid.MultiGridEnv):
    """Single-agent maze environment specified via a bit map."""

    def __init__(
        self,
        agent_view_size=5,
        minigrid_mode=True,
        max_steps=None,
        start_pos=None,
        goal_pos=None,
        size=15,
        seed=None,
    ):
        self.seed(seed)

        self.size = size

        self._sample_start_and_goal_pos()

        if max_steps is None:
            max_steps = 2 * size * size

        super().__init__(
            n_agents=1,
            grid_size=size,
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            see_through_walls=True,  # Set this to True for maximum speed
            minigrid_mode=minigrid_mode,
        )

    def _sample_start_and_goal_pos(self):
        size = self.size
        top_left = (1, 1)
        top_right = (size - 2, 1)
        bottom_left = (1, size - 2)
        bottom_right = (size - 2, size - 2)
        choices = [top_left, top_right, bottom_left, bottom_right]

        agent_idx, goal_idx = self.np_random.choice(
            range(len(choices)), size=(2,), replace=False
        )
        agent_pos = choices[agent_idx]
        goal_pos = choices[goal_idx]

        self.start_pos = np.array(agent_pos)
        self.goal_pos = np.array(goal_pos)

    def _gen_maze(self, width, height):
        # Use Kruskal's to compute a MST with random edges (walls)
        # connecting a grid of cells
        assert (width - 2) % 2 == 1 and (height - 2) % 2 == 1, "Dimensions must be 2n+1"

        self._sample_start_and_goal_pos()

        h, w = (height - 2) // 2 + 1, (width - 2) // 2 + 1
        g = networkx.grid_graph([h, w])

        bit_map = np.ones((width - 2, height - 2))

        ds = UnionFind()  # track connected components

        for v in g.nodes:
            y, x = v[0], v[1]
            bit_map[y * 2][x * 2] = 0
            ds.add(v)

        # Randomly sample edge
        edges = list(g.edges)
        self.np_random.shuffle(edges)

        for u, v in edges:
            # convert u,v to full bitmap coordinates
            if not ds.connected(u, v):
                y1, x1 = u[0] * 2, u[1] * 2
                y2, x2 = v[0] * 2, v[1] * 2

                wall_y = y1 + (y2 - y1) // 2
                wall_x = x1 + (x2 - x1) // 2

                bit_map[wall_y][wall_x] = 0

                ds.union(u, v)

        self.bit_map = bit_map

        return bit_map

    def _gen_grid(self, width, height):
        self._gen_maze(width, height)

        # Create an empty grid
        self.grid = multigrid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Goal
        self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])

        # Agent
        self.place_agent_at_pos(0, self.start_pos)

        # Walls
        for x in range(self.bit_map.shape[0]):
            for y in range(self.bit_map.shape[1]):
                if self.bit_map[y, x]:
                    # Add an offset of 1 for the outer walls
                    self.put_obj(minigrid.Wall(), x + 1, y + 1)


class PerfectMazeSmall(MSTMazeEnv):
    """A 11x11 Maze environment."""

    def __init__(self, seed=None):
        super().__init__(size=11, seed=seed)


class PerfectMazeMedium(MSTMazeEnv):
    """A 11x11 Maze environment."""

    def __init__(self, seed=None):
        super().__init__(size=21, seed=seed)


class PerfectMazeLarge(MSTMazeEnv):
    """A 11x11 Maze environment."""

    def __init__(self, seed=None):
        super().__init__(size=51, seed=seed)


class PerfectMazeXL(MSTMazeEnv):
    """A 11x11 Maze environment."""

    def __init__(self, seed=None):
        super().__init__(size=101, seed=seed)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname


register.register(
    id="MultiGrid-PerfectMazeSmall-v0", entry_point=module_path + ":PerfectMazeSmall"
)

register.register(
    id="MultiGrid-PerfectMazeMedium-v0", entry_point=module_path + ":PerfectMazeMedium"
)

register.register(
    id="MultiGrid-PerfectMazeLarge-v0", entry_point=module_path + ":PerfectMazeLarge"
)

register.register(
    id="MultiGrid-PerfectMazeXL-v0", entry_point=module_path + ":PerfectMazeXL"
)
