# Copyright (c) 2020 Flowers Team
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import numpy as np
import pickle
import copy
from teachDeepRL.teachers.algos.riac import RIAC
from teachDeepRL.teachers.algos.alp_gmm import ALPGMM
from teachDeepRL.teachers.algos.covar_gmm import CovarGMM
from teachDeepRL.teachers.algos.random_teacher import RandomTeacher
from teachDeepRL.teachers.algos.oracle_teacher import OracleTeacher
from teachDeepRL.teachers.utils.test_utils import get_test_set_name
from collections import OrderedDict

def param_vec_to_param_dict(param_env_bounds, param):
    param_dict = OrderedDict()
    cpt = 0
    for i,(name, bounds) in enumerate(param_env_bounds.items()):
        if len(bounds) == 2:
            param_dict[name] = param[i]
            cpt += 1
        elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
            nb_dims = bounds[2]
            param_dict[name] = param[i:i+nb_dims]
            cpt += nb_dims
    #print('reconstructed param vector {}\n into {}'.format(param, param_dict)) #todo remove
    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):  # needs param_env_bounds for order reference
    param_vec = []
    for name, bounds in param_env_bounds.items():
        #print(param_dict[name])
        param_vec.append(param_dict[name])
    return np.array(param_vec, dtype=np.float32)



class TeacherController(object):
    def __init__(self, teacher, nb_test_episodes, param_env_bounds, reward_bounds=None, seed=None, teacher_params={}):
        self.teacher = teacher
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)
        self.reward_bounds = reward_bounds

        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
            if len(bounds) == 2:
                mins.append(bounds[0])
                maxs.append(bounds[1])
            elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                mins.extend([bounds[0]] * bounds[2])
                maxs.extend([bounds[1]] * bounds[2])
            else:
                print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                exit(1)

        # setup tasks generator
        if teacher == 'Oracle':
            self.task_generator = OracleTeacher(mins, maxs, teacher_params['window_step_vector'], seed=seed)
        elif teacher == 'Random':
            self.task_generator = RandomTeacher(mins, maxs, seed=seed)
        elif teacher == 'RIAC':
            self.task_generator = RIAC(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'ALP-GMM':
            self.task_generator = ALPGMM(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'Covar-GMM':
            self.task_generator = CovarGMM(mins, maxs, seed=seed, params=teacher_params)
        else:
            print('Unknown teacher')
            raise NotImplementedError

        #data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []

    def record_train_episode(self, reward, index=0):
        self.env_train_rewards.append(reward)
        if self.teacher != 'Oracle':
            if self.reward_bounds:
                reward = np.interp(reward, self.reward_bounds, (0, 1))
            self.env_train_norm_rewards.append(reward)
            
        self.task_generator.update(self.env_params_train[index], reward)

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_param_bounds': list(self.param_env_bounds.items())}
            dump_dict = self.task_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_env_params(self, env):
        batch_params = []
        for _ in range(env.num_envs):
            batch_params.append(copy.copy(self.task_generator.sample_task()))
        assert type(batch_params[0][0]) == np.float32

        batch_param_dict = []
        for params in batch_params:
            param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
            batch_param_dict.append(param_dict)

        self.env_params_train = batch_params

        obs = env.reset_alp_gmm(batch_params)

        return obs