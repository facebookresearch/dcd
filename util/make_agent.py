# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from algos import PPO, RolloutStorage, ACAgent
from models import \
    MultigridNetwork, MultigridGlobalCriticNetwork, \
    CarRacingNetwork, \
    CarRacingBezierAdversaryEnvNetwork, \
    BipedalWalkerStudentPolicy, \
    BipedalWalkerAdversaryPolicy

def model_for_multigrid_agent(
    env,
    agent_type='agent',
    recurrent_arch=None,
    recurrent_hidden_size=256,
    use_global_critic=False,
    use_global_policy=False):
    if agent_type == 'adversary_env':
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space
        adversary_max_timestep = adversary_observation_space['time_step'].high[0] + 1
        adversary_random_z_dim = adversary_observation_space['random_z'].shape[0]

        model = MultigridNetwork(
            observation_space=adversary_observation_space, 
            action_space=adversary_action_space,
            conv_filters=128,
            scalar_fc=10,
            scalar_dim=adversary_max_timestep,
            random_z_dim=adversary_random_z_dim,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_directions = observation_space['direction'].high[0] + 1 
        model_kwargs = dict(
            observation_space=observation_space, 
            action_space=action_space,
            scalar_fc=5,
            scalar_dim=num_directions,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)

        model_constructor = MultigridNetwork
        if use_global_critic:
            model_constructor = MultigridGlobalCriticNetwork

        if use_global_policy:
            model_kwargs.update({'use_global_policy': True})

        model = model_constructor(**model_kwargs)

    return model

def model_for_car_racing_agent(
    env,
    agent_type='agent',
    use_skip=False,
    choose_start_pos=False,
    use_popart=False,
    adv_use_popart=False,
    use_categorical_adv=False,
    use_goal=False,
    num_goal_bins=1):
    if agent_type == 'adversary_env':
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space
        model = CarRacingBezierAdversaryEnvNetwork(
            observation_space=adversary_observation_space,
            action_space=adversary_action_space,
            use_categorical=use_categorical_adv,
            use_skip=use_skip,
            choose_start_pos=choose_start_pos,
            use_popart=adv_use_popart,
            use_goal=use_goal,
            num_goal_bins=num_goal_bins)
    else:
        action_space = env.action_space
        obs_shape = env.observation_space.shape
        model = CarRacingNetwork(
            obs_shape=obs_shape,
            action_space = action_space,
            hidden_size=100,
            use_popart=use_popart) 

    return model

def model_for_bipedalwalker_agent(
    env,
    agent_type='agent',
    recurrent_arch=False):
    if 'adversary_env' in agent_type:
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space

        model = BipedalWalkerAdversaryPolicy(
                observation_space=adversary_observation_space,
                action_space=adversary_action_space)

    else:
        model = BipedalWalkerStudentPolicy(
            obs_shape=env.observation_space.shape,
            action_space=env.action_space,
            recurrent=recurrent_arch)

    return model

def model_for_env_agent(
    env_name,
    env,
    agent_type='agent',
    recurrent_arch=None,
    recurrent_hidden_size=256,
    use_global_critic=False,
    use_global_policy=False,
    use_skip=False,
    choose_start_pos=False,
    use_popart=False,
    adv_use_popart=False,
    use_categorical_adv=False,
    use_goal=False,
    num_goal_bins=1):
    assert agent_type in ['agent', 'adversary_agent', 'adversary_env']
        
    if env_name.startswith('MultiGrid'):
        model = model_for_multigrid_agent(
            env=env, 
            agent_type=agent_type,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size,
            use_global_critic=use_global_critic,
            use_global_policy=use_global_policy)
    elif env_name.startswith('CarRacing'):
        model = model_for_car_racing_agent(
            env=env, 
            agent_type=agent_type,
            use_skip=use_skip,
            choose_start_pos=choose_start_pos,
            use_popart=use_popart,
            adv_use_popart=adv_use_popart,
            use_categorical_adv=use_categorical_adv,
            use_goal=use_goal,
            num_goal_bins=num_goal_bins)
    elif env_name.startswith('BipedalWalker'):
        model = model_for_bipedalwalker_agent(
            env=env,
            agent_type=agent_type,
            recurrent_arch=recurrent_arch)
    else:
        raise ValueError(f'Unsupported environment {env_name}.')

    return model


def make_agent(name, env, args, device='cpu'):
    # Create model instance
    is_adversary_env = 'env' in name

    if is_adversary_env:
        observation_space = env.adversary_observation_space
        action_space = env.adversary_action_space
        num_steps = observation_space['time_step'].high[0]
        recurrent_arch = args.recurrent_adversary_env and args.recurrent_arch
        entropy_coef = args.adv_entropy_coef
        ppo_epoch = args.adv_ppo_epoch
        num_mini_batch = args.adv_num_mini_batch
        max_grad_norm = args.adv_max_grad_norm
        use_popart = vars(args).get('adv_use_popart', False)
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_steps = args.num_steps
        recurrent_arch = args.recurrent_agent and args.recurrent_arch
        entropy_coef = args.entropy_coef
        ppo_epoch = args.ppo_epoch
        num_mini_batch = args.num_mini_batch
        max_grad_norm = args.max_grad_norm
        use_popart = vars(args).get('use_popart', False)

    recurrent_hidden_size = args.recurrent_hidden_size

    actor_critic = model_for_env_agent(
        args.env_name, env, name, 
        recurrent_arch=recurrent_arch,
        recurrent_hidden_size=recurrent_hidden_size,
        use_global_critic=args.use_global_critic,
        use_global_policy=vars(args).get('use_global_policy', False),
        use_skip=vars(args).get('use_skip', False),
        choose_start_pos=vars(args).get('choose_start_pos', False),
        use_popart=vars(args).get('use_popart', False),
        adv_use_popart=vars(args).get('adv_use_popart', False),
        use_categorical_adv=vars(args).get('use_categorical_adv', False),
        use_goal=vars(args).get('sparse_rewards', False),
        num_goal_bins=vars(args).get('num_goal_bins', 1))

    algo = None
    storage = None
    agent = None

    use_proper_time_limits = \
        hasattr(env, 'get_max_episode_steps') \
        and env.get_max_episode_steps() is not None \
        and vars(args).get('handle_timelimits', False)

    if args.algo == 'ppo':
        # Create PPO
        algo = PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=max_grad_norm,
            clip_value_loss=args.clip_value_loss,
            log_grad_norm=args.log_grad_norm
        )

        # Create storage
        storage = RolloutStorage(
            model=actor_critic,
            num_steps=num_steps,
            num_processes=args.num_processes,
            observation_space=observation_space,
            action_space=action_space,
            recurrent_hidden_state_size=args.recurrent_hidden_size,
            recurrent_arch=args.recurrent_arch,
            use_proper_time_limits=use_proper_time_limits,
            use_popart=use_popart
        )

        agent = ACAgent(algo=algo, storage=storage).to(device)

    else:
        raise ValueError(f'Unsupported RL algorithm {algo}.')

    return agent
