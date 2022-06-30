# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class RaceTrack(object):
	def __init__(self, 
				 name, 
				 xy, 
				 bounds, 
				 full_zoom=0.12,
				 max_episode_steps=1000):
		self.name = name
		self.xy = xy
		self.bounds = bounds
		self.full_zoom = full_zoom
		self.max_episode_steps = max_episode_steps