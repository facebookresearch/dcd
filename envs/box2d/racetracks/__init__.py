# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .formula1 import *

def get_track(name):
	if isinstance(name, str):
		return getattr(sys.modules[__name__], name, None)
	else:
		return None