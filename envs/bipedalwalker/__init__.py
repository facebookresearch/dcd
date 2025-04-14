# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .adversarial import BipedalWalkerAdversarialEnv
from .walker_test_envs import BipedalWalkerDefault
import pandas as pd


BIPEDALWALKER_POET_DF_COLUMNS = [
    "roughness",
    "pitgap_low",
    "pitgap_high",
    "stumpheight_low",
    "stumpheight_high",
    "seed",
]

BIPEDALWALKER_DF_COLUMNS = [
    "roughness",
    "pitgap_low",
    "pitgap_high",
    "stumpheight_low",
    "stumpheight_high",
    "stairheight_low",
    "stairheight_high",
    "stair_steps",
    "seed",
]


def bipedalwalker_df_from_encodings(env_name, encodings):
    df = pd.DataFrame(encodings)
    if "POET" in env_name:
        df.columns = BIPEDALWALKER_POET_DF_COLUMNS
    else:
        df.columns = BIPEDALWALKER_DF_COLUMNS

    return df
