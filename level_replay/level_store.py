# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple, defaultdict, deque

import numpy as np
import torch


INT32_MAX = 2147483647


class LevelStore(object):
	"""
	Manages a mapping between level index --> level, where the level
	may be represented by any arbitrary data structure. Typically, we can 
	represent any given level as a string.
	"""
	def __init__(self, max_size=None, data_info={}):
		self.max_size = max_size
		self.seed2level = defaultdict()
		self.level2seed = defaultdict()
		self.seed2parent = defaultdict()
		self.next_seed = 1
		self.levels = set()

		self.data_info = data_info

	def __len__(self):
		return len(self.levels)

	def _insert(self, level, parent_seed=None):
		if level is None:
			return None

		if level not in self.levels:
			# FIFO if max size constraint
			if self.max_size is not None:
				while len(self.levels) >= self.max_size:
					first_idx = list(self.seed2level)[0]
					self._remove(first_idx)

			seed = self.next_seed
			self.seed2level[seed] = level
			if parent_seed is not None:
				self.seed2parent[seed] = \
					self.seed2parent[parent_seed] + [self.seed2level[parent_seed]]
			else:
				self.seed2parent[seed] = []
			self.level2seed[level] = seed
			self.levels.add(level)
			self.next_seed += 1
			return seed
		else:
			return self.level2seed[level]

	def insert(self, level, parent_seeds=None):
		if hasattr(level, '__iter__'):
			idx = []
			for i, l in enumerate(level):
				ps = None
				if parent_seeds is not None:
					ps = parent_seeds[i]
				idx.append(self._insert(l, ps))
			return idx
		else:
			return self._insert(level) 

	def _remove(self, level_seed):
		if level_seed is None or level_seed < 0:
			return

		level = self.seed2level[level_seed]
		self.levels.remove(level)
		del self.seed2level[level_seed]
		del self.level2seed[level]
		del self.seed2parent[level_seed]

	def remove(self, level_seed):
		if hasattr(level_seed, '__iter__'):
			for i in level_seed:
				self._remove(i)
		else:
			self._remove(level_seed)

	def reconcile_seeds(self, level_seeds):
		old_seeds = set(self.seed2level)
		new_seeds = set(level_seeds)

		# Don't update if empty seeds
		if len(new_seeds) == 1 and -1 in new_seeds:
			return

		ejected_seeds = old_seeds - new_seeds
		for seed in ejected_seeds:
			self._remove(seed)

	def get_level(self, level_seed):
		level = self.seed2level[level_seed]

		if self.data_info:
			if self.data_info.get('numpy', False):
				dtype = self.data_info['dtype']
				shape = self.data_info['shape']
				level = np.frombuffer(level, dtype=dtype).reshape(*shape)

		return level
