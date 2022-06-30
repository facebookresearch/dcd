# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from envs.registration import register as gym_register

from .racetracks import RaceTrack
from .racetracks import formula1
from .car_racing_bezier import CarRacingBezier


def set_global(name, value):
    globals()[name] = value


racetracks = dict([(name, cls) for name, cls in formula1.__dict__.items() if isinstance(cls, RaceTrack)])


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


def _create_constructor(track):
	def constructor(self, **kwargs):
		return CarRacingBezier.__init__(self, 
			track_name=track.name,
			**kwargs)
	return constructor


for name, track in racetracks.items():
	class_name = f"CarRacingF1-{track.name}"
	env = type(class_name, (CarRacingBezier, ), {
	    "__init__": _create_constructor(track),
	})
	set_global(class_name, env)
	gym_register(
		id=f'CarRacingF1-{track.name}-v0', 
		entry_point=module_path + f':{class_name}',
	    max_episode_steps=track.max_episode_steps,
	    reward_threshold=900)