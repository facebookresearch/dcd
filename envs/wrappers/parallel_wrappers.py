# Copyright (c) 2019 Antonin Raffin
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a heavily modified version of
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py

import multiprocessing as mp

import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper
from baselines.common.vec_env.vec_env import clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrappers):
    def step(env, action):
        ob, reward, done, info = env.step(action)

        if done:
            ob = env.reset()
        return ob, reward, done, info

    def step_env(env, action, reset_random=False):
        ob, reward, done, info = env.step(action)

        if done:
            if reset_random:
                env.reset_random()
                ob = env.reset_agent()
            else:
                ob = env.reset_agent()

        return ob, reward, done, info

    def get_env_attr(env, attr):
        if hasattr(env, attr):
            return getattr(env, attr)

        while hasattr(env, 'env'):
            env = env.env
            if hasattr(env, attr):
                return getattr(env, attr)

        return None

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step(env, action) for env, action in zip(envs, data)])
            elif cmd == 'step_env':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'step_env_reset_random':
                remote.send([step_env(env, action, reset_random=True) for env, action in zip(envs, data)])
            elif cmd == 'observation_space':
                remote.send(envs[0].observation_space)
            elif cmd == 'adversary_observation_space':
                remote.send(envs[0].adversary_observation_space)
            elif cmd == 'adversary_action_space':
                remote.send(envs[0].adversary_action_space)
            elif cmd == 'max_steps':
                remote.send(envs[0].max_steps)
            elif cmd == 'render':
                remote.send([env.render(mode='level') for env in envs])
            elif cmd == 'render_to_screen':
                remote.send([envs[0].render('human')])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            elif cmd == 'reset_to_level':
                remote.send([envs[0].reset_to_level(data)])
            elif cmd == 'reset_alp_gmm':
                remote.send([envs[0].reset_alp_gmm(data)])
            elif cmd == 'max_episode_steps':
                max_episode_steps = get_env_attr(envs[0], '_max_episode_steps')
                remote.send(max_episode_steps)
            elif hasattr(envs[0], cmd):
                attrs = [getattr(env, cmd) for env in envs]
                is_callable = hasattr(attrs[0], '__call__')
                if is_callable:
                    if not hasattr(data, '__len__'):
                        data = [data]*len(attrs)
                    remote.send([attr(d) if d is not None else attr() for attr, d in zip(attrs, data)])
                else:
                    remote.send([attr for attr in attrs])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1, is_eval=False):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

        # Get processed action dim
        self.is_eval = is_eval
        self.processed_action_dim = 1
        if not is_eval:
            self.remotes[0].send(('processed_action_dim', None))
            self.processed_action_dim = self.remotes[0].recv()[0]

    def step_async(self, action):
        self._assert_not_closed()
        action = np.array_split(action, self.nremotes)
        for remote, action in zip(self.remotes, action):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_complexity_info(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_complexity_info', None))
        info = [remote.recv() for remote in self.remotes]
        info = _flatten_list(info)
        return info

    def get_images(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('render', None))
        imgs = [remote.recv() for remote in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def render_to_screen(self):
        self._assert_not_closed()
        self.remotes[0].send(('render_to_screen', None))
        return self.remotes[0].recv()

    def max_episode_steps(self):
        self._assert_not_closed()
        self.remotes[0].send(('max_episode_steps', None))
        return self.remotes[0].recv()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


class ParallelAdversarialVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, adversary=True, is_eval=False):
        super().__init__(env_fns, is_eval=is_eval)
        action_space = self.action_space
        if action_space.__class__.__name__ == 'Box':
            self.action_dim = action_space.shape[0]
        else:
            self.action_dim = 1

        self.adv_action_dim = 0
        if adversary:
            adv_action_space = self.adversary_action_space
            if adv_action_space.__class__.__name__ == 'Box':
                self.adv_action_dim = adv_action_space.shape[0]
            else:
                self.adv_action_dim = 1

    def _should_expand_action(self, action, adversary=False):
        if not adversary:
            action_dim = self.action_dim
        else:
            action_dim = self.adv_action_dim
        # print('expanding actions?', action_dim>1, flush=True)
        return action_dim > 1 or self.processed_action_dim > 1

    def seed_async(self, seed, index):
        self._assert_not_closed()
        self.remotes[index].send(('seed', seed))
        self.waiting = True

    def seed_wait(self, index):
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def seed(self, seed, index):
        self.seed_async(seed, index)
        return self.seed_wait(index)

    def level_seed_async(self, index):
        self._assert_not_closed()
        self.remotes[index].send(('level_seed', None))
        self.waiting = True

    def level_seed_wait(self, index):
        self._assert_not_closed()
        level_seed = self.remotes[index].recv()
        self.waiting = False
        return level_seed

    def level_seed(self, index):
        self.level_seed_async(index)
        return self.level_seed_wait(index)

    # step_adversary
    def step_adversary(self, action):
        if self._should_expand_action(action, adversary=True):
            action = np.expand_dims(action, 1)
        self.step_adversary_async(action)
        return self.step_wait()

    def step_adversary_async(self, action):
        self._assert_not_closed()
        [remote.send(('step_adversary', a)) for remote, a in zip(self.remotes, action)]
        self.waiting = True

    def step_env_async(self, action):
        self._assert_not_closed()
        if self._should_expand_action(action):
            action = np.expand_dims(action, 1)
        [remote.send(('step_env', a)) for remote, a in zip(self.remotes, action)]
        self.waiting = True

    def step_env_reset_random_async(self, action):
        self._assert_not_closed()
        if self._should_expand_action(action):
            action = np.expand_dims(action, 1)
        [remote.send(('step_env_reset_random', a)) for remote, a in zip(self.remotes, action)]
        self.waiting = True

    # reset_agent
    def reset_agent(self):
        self._assert_not_closed()
        [remote.send(('reset_agent', None)) for remote in self.remotes]
        self.waiting = True
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # reset_random
    def reset_random(self):
        self._assert_not_closed()
        [remote.send(('reset_random', None)) for remote in self.remotes]
        self.waiting = True
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # reset_to_level
    def reset_to_level(self, level, index):
        self._assert_not_closed()
        self.remotes[index].send(('reset_to_level', level))
        self.waiting = True
        obs = self.remotes[index].recv()
        self.waiting = False
        return _flatten_obs(obs)

    def reset_to_level_batch(self, level):
        self._assert_not_closed()
        [remote.send(('reset_to_level', level[i])) for i, remote in enumerate(self.remotes)]
        self.waiting = True
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # mutate level
    def mutate_level(self, num_edits):
        self._assert_not_closed()
        [remote.send(('mutate_level', num_edits)) for _, remote in enumerate(self.remotes)]
        self.waiting = True
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # observation_space
    def get_observation_space(self):
        self._assert_not_closed()
        self.remotes[0].send(('observation_space', None))
        obs_space = self.remotes[0].recv()
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces
        return obs_space

    # adversary_observation_space
    def get_adversary_observation_space(self):
        self._assert_not_closed()
        self.remotes[0].send(('adversary_observation_space', None))
        obs_space = self.remotes[0].recv()
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces
        return obs_space

    def get_adversary_action_space(self):
        self._assert_not_closed()
        self.remotes[0].send(('adversary_action_space', None))
        action_dim = self.remotes[0].recv()
        return action_dim

    def get_max_episode_steps(self):
        self._assert_not_closed()
        self.remotes[0].send(('max_episode_steps', None))
        self.waiting = True
        max_episode_steps = self.remotes[0].recv()
        self.waiting = False
        return max_episode_steps

    # Generic getter
    def remote_attr(self, name, data=None, flatten=False, index=None):
        self._assert_not_closed()

        if index is None or len(index) == 0:
            remotes = self.remotes
        else:
            remotes = [self.remotes[i] for i in index]

        if hasattr(data, '__len__'):
            assert len(data) == len(remotes)
            [remote.send((name, d)) for remote, d in zip(remotes, data)]
        else:
            [remote.send((name, data)) for remote in remotes]
        self.waiting = True
        result = [remote.recv() for remote in remotes]
        self.waiting = False
        return _flatten_list(result) if flatten else result

    def get_seed(self):
        return self.remote_attr('seed_value', flatten=True)

    def set_seed(self, seeds):
        return self.remote_attr('seed', data=seeds, flatten=True)

    def get_level(self):
        levels = self.remote_attr('level')
        return [l[0] for l in levels]  # flatten

    def get_encodings(self, index=None):
        return self.remote_attr('encoding', flatten=True, index=index)

    # Navigation-specific
    def get_distance_to_goal(self):
        return self.remote_attr('distance_to_goal', flatten=True)

    def get_passable(self):
        return self.remote_attr('passable', flatten=True)

    def get_shortest_path_length(self):
        return self.remote_attr('shortest_path_length', flatten=True)

    # ALP-GMM-specific
    def reset_alp_gmm(self, levels):
        self._assert_not_closed()
        [remote.send(('reset_alp_gmm', levels[i])) for i, remote in enumerate(self.remotes)]
        self.waiting = True
        self._assert_not_closed()
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    # === Multigrid-specific ===
    def get_num_blocks(self):
        return self.remote_attr('n_clutter_placed', flatten=True)

    def __getattr__(self, name):
        if name == 'observation_space':
            return self.get_observation_space()
        elif name == 'adversary_observation_space':
            return self.get_adversary_observation_space()
        elif name == 'adversary_action_space':
            return self.get_adversary_action_space()
        elif name == 'max_steps':
            return self.get_max_steps()
        else:
            return self.__getattribute__(name)