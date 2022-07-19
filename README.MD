# Dual Curriculum Design

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

![DCD overview diagram](/docs/images/dcd_overview.png#gh-light-mode-only)
![DCD overview diagram](/docs/images/dcd_overview_darkmode.png#gh-dark-mode-only)

This codebase contains an extensible framework for implementing various Unsupervised Environment Design (UED) algorithms, including state-of-the-art Dual Curriculum Design (DCD) algorithms with minimax-regret robustness properties like ACCEL and Robust PLR:
- [ACCEL](https://accelagent.github.io/)
- [Robust PLR](https://arxiv.org/abs/2110.02439)
- [PLR](https://arxiv.org/abs/2010.03934)
- [REPAIRED](https://arxiv.org/abs/2110.02439)
- [PAIRED](https://arxiv.org/abs/2012.02096)
- [ALP-GMM](https://arxiv.org/abs/1910.07224)
- [Minimax adversarial training](https://arxiv.org/abs/2012.02096)
- [Domain randomization (DR)](https://arxiv.org/abs/1703.06907)

We also include experiment configurations for the main experiments in the following papers on DCD methods:
- [Replay-Guided Adversarial Design. Jiang et al, 2021 (NeurIPS 2021)](https://arxiv.org/abs/2110.02439)
- [Evolving Curricula with Regret-Based Environment Design. Parker-Holder et al, 2022 (ICML 2022)](https://accelagent.github.io/)

## Citation
The core components of this codebase, as well as the `CarRacingBezier` and `CarRacingF1` environments, were introduced in [Jiang et al, 2021](https://arxiv.org/abs/2110.02439). If you use this code to develop your own UED algorithms in academic contexts, please cite 

    Jiang et al, "Replay-Guided Adversarial Environment Design", 2021.
([Bibtex here](https://raw.githubusercontent.com/facebookresearch/dcd/main/docs/bibtex/dcd.bib))

Additionally, if you use ACCEL or the adversarial `BipedalWalker` environments in academic contexts, please cite 
    
    Parker-Holder et al, "Evolving Curricula with Regret-Based Environment Design", 2022.
([Bibtex here](https://raw.githubusercontent.com/facebookresearch/dcd/main/docs/bibtex/accel.bib))

## Setup
To install the necessary dependencies, run the following commands:
```
conda create --name dcd python=3.8
conda activate dcd
pip install -r requirements.txt
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install pyglet==1.5.11
```

## A quick overview of [`train.py`](https://github.com/facebookresearch/dcd/blob/master/train.py)

### Choosing a UED algorithm
The exact UED algorithm is specified by a combination of values for `--ued_algo`, `--use_plr`, `no_exploratory_grad_updates`, and  `--ued_editor`:
| Method        | `ued_algo`  | `use_plr`| `no_exploratory_grad_updates` | `ued_editor`| 
| ------------- |:-------------|:-------------|:-------------|:-------------|
| DR | `domain_randomization` | `false` | `false` | `false` |
| PLR | `domain_randomization` | `true` | `false`  | `false` |
| PLR<sup>⊥</sup> | `domain_randomization` | `true` | `true` | `false` |
| ACCEL | `domain_randomization` | `true` | `true` | `true` |
| PAIRED | `paired` | `false` | `false` | `false` |
| REPAIRED | `paired` | `true` | `true`  | `false` |
| Minimax | `minimax` | `false` | `false` | `false` |


Full details for the command-line arguments related to PLR and ACCEL can be found in [`arguments.py`](https://github.com/facebookresearch/dcd/blob/master/arguments.py). We provide simple configuration JSON files for generating the `train.py` commands for the best hyperparameters found in experimental settings from prior works.

### Logging
By default, `train.py` generates a folder in the directory specified by the `--log_dir` argument, named according to `--xpid`. This folder contains the main training logs, `logs.csv`, and periodic screenshots of generated levels in the directory `screenshots`. Each screenshot uses the naming convention `update_<number of PPO updates>.png`. When ACCEL is turned on, the screenshot naming convention also includes information about whether the level was replayed via PLR and the mutation generation number for the level, i.e. how many mutation cycles led to this level.

### Checkpointing
**Latest checkpoint**
The latest model checkpoint is saved as `model.tar`. The model is checkpointed every `--checkpoint_interval` number of updates. When setting `--checkpoint_basis=num_updates` (default), the checkpoint interval corresponds to number of rollout cycles (which includes one rollout for each student and teacher). Otherwise, when `--checkpoint_basis=student_grad_updates`, the checkpoint interval corresponds to the number of PPO updates performed by the student agent only. This latter checkpoint basis allows comparing methods based on number of gradient updates actually performed by the student agent, which can differ from number of rollout cycles, as methods based on Robust PLR, like ACCEL, do not perform student gradient updates every rollout cycle.

**Archived checkpoints**
Separate archived model checkpoints can be saved at specific intervals by specifying a positive value for the argument `--archive_interval`. For example, setting `--archive_interval=1250` and `--checkpoint_basis=student_grad_updates` will result in saving model checkpoints named `model_1250.tar`, `model_2500.tar`, and so on. These archived models are saved in addition to `model.tar`, which always stores the latest checkpoint, based on `--checkpoint_interval`.

## Evaluating agents with [`eval.py`](https://github.com/facebookresearch/dcd/blob/master/eval.py)

### Evaluating a single model
The following command evaluates a `<model>.tar` in an experiment results directory, `<xpid>`, in a base log output directory `<log_dir>` for `<num_episodes>` episodes in each of the environments named `<env_name1>`, `<env_name1>`, and `<env_name1>`, and outputs the results as a .csv in `<result_dir>`.
```shell
python -m eval \
--base_path <log_dir> \
--xpid <xpid> \
--model_tar <model>
--env_names <env_name1>,<env_name2>,<env_name3> \
--num_episodes <num_episodes> \
--result_path <result_dir>
```

### Evaluating multiple models
Similarly, the following command evaluates all models named `<model>.tar` in experiment results directories matching the prefix `<xpid_prefix>`. This prefix argument is useful for evaluating models from a set of training runs with the same hyperparameter settings. The resulting .csv will contain a column for each model matched and evaluated this way.
```shell
python -m eval \
--base_path <log_dir> \
--prefix <xpid_prefix> \
--model_tar <model>
--env_names <env_name1>,<env_name2>,<env_name3> \
--num_episodes <num_episodes> \
--result_path <result_dir>
```

### Evaluating on zero-shot benchmarks
Replacing the `--env_names=...` argument with the `--benchmark=<benchmark>` argument will perform evaluation over a set of benchmark test environments for the domain specified by `<benchmark>`. The various zero-shot benchmarks are described below:
| `benchmark`        | Description  |
| ------------- |:-------------|
| `maze` | Human-designed mazes, including singleton and procedurally-generated designs. | 
| `f1` | The full `CarRacing-F1` benchmark: 20 challenging tracks from the Formula-1. |
| `bipedal` | `BipedalWalker-v3`, `BipedalWalkerHardcore-v3`, and isolated challenges for stairs, stumps, pit gaps, and ground roughness.  |
| `poetrose` | Environments based on the most extremely challenging level settings discovered by POET, as reported in the red polygons in the top two rows of Figure 5 in [Wang et al, 2019](https://arxiv.org/abs/1901.01753). |


## Running experiments
We provide configuration json files to generate the `train.py` commands for the specific experiment settings featured in the main results of previous works. To generate the command to launch 1 run of the experiment described by the configuration file `config.json` in the folder `train_scripts/grid_configs`, simply run the following, and copy and paste the output into your command line. 
```shell
python train_scripts/make_cmd.py --json config --num_trials 1
```

Alternatively, you can run the following to copy the command directly to your clipboard:

```shell
python train_scripts/make_cmd.py --json config --num_trials 1 | pbcopy
```

The JSON files for training methods using the best hyperparameters settings in each environment are detailed below.

## Environments

### 🧭 MiniGrid Mazes

![Example mazes](/docs/images/ood_mazes.png)

The [MiniGrid-based mazes](https://github.com/facebookresearch/dcd/tree/master/envs/multigrid) from [Dennis et al, 2020](https://arxiv.org/abs/2012.02096) and [Jiang et al, 2021](https://arxiv.org/abs/2110.02439) require agents to perform partially-observable navigation. Various human-designed singleton and procedurally-generated mazes allow testing of zero-shot transfer performance to out-of-distribution configurations.

#### Experiments from [Jiang et al, 2021](https://arxiv.org/abs/2110.02439)
| Method        | json config  |
| ------------- |:-------------|
| PLR<sup>⊥</sup> | `minigrid/25_blocks/mg_25b_robust_plr.json`| 
| PLR| `minigrid/25_blocks/mg_25b_plr.json` |
| REPAIRED| `minigrid/25_blocks/mg_25b_repaired.json`|
| Minimax | `minigrid/25_blocks/mg_25b_minimax.json`|
| DR | `minigrid/25_blocks/mg_25b_dr.json`|

#### Experiments from [Parker-Holder et al, 2022](https://accelagent.github.io/)
| Method        | json config  |
| ------------- |:-------------|
| ACCEL (from empty) | `minigrid/60_blocks_uniform/mg_60b_uni_accel_empty.json`| 
| PLR<sup>⊥</sup> (Uniform(0-60) blocks) | `minigrid/mg_60b_uni_robust_plr.json` |
| DR (Uniform(0-60) blocks) | `minigrid/mg_60b_uni_dr.json`|

### 🏎 CarRacing

#### `CarRacingBezier`

![CarRacingBezier tracks](/docs/images/car_racing_bezier_tracks.png)

The [`CarRacingBezier`](https://github.com/facebookresearch/dcd/tree/master/envs/box2d) environment introduced in [Jiang et al, 2021](https://arxiv.org/abs/2110.02439) reparameterizes the tracks in the original `CarRacing` environment from OpenAI Gym using Bézier curves. By default, `CarRacingBezier` generates tracks by randomizing 12 control points defining a Bézier curve.

#### `CarRacingF1`
![CarRacingF1 tracks](/docs/images/f1_benchmark_tracks.png)
The [`CarRacingF1`](https://github.com/facebookresearch/dcd/tree/master/envs/box2d/racetracks) benchmark introduced in [Jiang et al, 2021](https://arxiv.org/abs/2110.02439) consists of 20 challenging tracks based on the official Formula-1 tracks. This benchmark allows evaluation of zero-shot transfer performance to longer tracks with more difficult turns.
| Method        | json config  |
| ------------- |:-------------|
| PLR<sup>⊥</sup> | `car_racing/cr_robust_plr.json`| 
| PLR| `car_racing/cr_plr.json` |
| REPAIRED| `car_racing/cr_repaired.json`|
| PAIRED | `car_racing/cr_paired.json`|
| DR | `car_racing/cr_dr.json`|

### 🦿🦿 BipedalWalker

![Example BipedalWalker environments](/docs/images/bipedal_challenges.png)

The [BipedalWalker environment](https://github.com/facebookresearch/dcd/tree/master/envs/bipedalwalker) requires continuous control of a 2D bipedal robot over challenging terrain with various obstacles, using a propriocetive observation. The zero-shot transfer configurations, used in [Parker-Holder et al, 2022](https://accelagent.github.io/), include `BipedalWalkerHardcore`, environments featuring each challenge (i.e. ground roughness, stump, pit gap, and stairs) in isolation, as well as extremely challenging configurations discovered by POET in [Wang et al, 2019](https://arxiv.org/abs/1901.01753).

| Method        | json config  |
| ------------- |:-------------|
| ACCEL | `bipedal/bipedal_accel.json`| 
| ACCEL (in POET design space) | `bipedal/bipedal_accel_poet.json`| 
| PLR<sup>⊥</sup> | `bipedal/bipedal_robust_plr.json` | 
| PAIRED | `bipedal/bipedal_paired.json`|
| Minimax | `bipedal/bipedal_minimax.json`|
| DR | `bipedal/bipedal_dr.json`|

![Example ACCEL BipedalWalker agent on challenging terrain](/docs/images/accel_bipedal_demo.gif)


### Current environment support
| Method        | 🧭 MiniGrid mazes | 🏎 CarRacing | 🦿🦿 BipedalWalker |
| ------------- |:-------------|:-------------|:-------------|
| ACCEL | ✅ | ❌ | ✅ |
| PLR<sup>⊥</sup> |✅ |✅ |✅ | 
| PLR  |✅ |✅ |✅ | 
| REPAIRED  |✅ |✅ |✅ | 
| PAIRED  |✅ |✅ |✅ | 
| Minimax  |✅ |✅ |✅ | 
| DR  |✅ |✅ |✅ | 


## 🏗 Integrating a new environment

### Adding your environment

1. Create a module for your new environment in [`envs`](https://github.com/facebookresearch/dcd/tree/master/envs).
2. Create an adversarial version of your environment. An easy approach is to define `<YourEnv>Adversarial` inside a file called `adversarial.py`.
3. Support the required functions inside `<YourEnv>Adversarial` for the UED algorithms of interest:

### Supporting domain randomization (DR)
- `reset_agent()`: Reset the current environment level, i.e. only the student agent's state. Do not change the actual environment configuration. Returns the first observation in a new episode starting in that level.
- `reset_random()`: Reset the environment to a random level, and return the observation for the first time step in this level.

### Supporting PAIRED
Your adversarial environment must support methods for allowing an adversarial teacher policy to incrementally design a level of the environment step-by-step.
- `reset_agent()`
- `reset()`: Reset the environment to an initial state. For example, for a maze environment, this initial state is an empty grid.
- `step_adversary(action)`: Transition the environment when the teacher performs <action>. Returns a Gym-style transition tuple to the teacher's policy, (`obs`, `reward`, `done`, `info_dict`), where `reward` is always 0, and `done=True` when environment design is completed. The `obs` is an observation dictionary or tensor that the teacher policy receives as input at the next step.

You must also define the adversary observation space and action space inside the `__init__.py` method of `<YourEnv>Adversarial`:
- `adversary_observation_space`: Gym space for the adversary's observation space.
- `adversary_action_space`: Gym space for the adversary's action space.

### Supporting PLR and PLR<sup>⊥</sup>
By default, the `domain_randomization` generator used by PLR will try to generate each new level using a uniformly random teacher policy, requiring the implementation of `step_adversary` as in PAIRED/REPAIRED. However, only `reset_random` is needed if the additional flag `--use_reset_random_dr=True` is passed to `train.py`.
- `reset_agent()`
- `reset()`
- Either `reset_random()` or `step_adversary(action)`
- `reset_to_level(level)`: This method receives a level encoding `<level>` and resets the environment to the corresponding level, and returns the observation for the first time step in this level. 

### Supporting REPAIRED
To support REPAIRED, implement the methods needed to support PAIRED and PLR.

### Supporting ACCEL
- `reset_agent()`
- `reset()`
- Either `reset_random()` or `step_adversary(action)`
- `mutate_level(num_edits)`: This method applies <num_edits> random edits to the current level to produce a mutation of it, resets the environment to this mutated level, and returns the observation for the first time step in this level.

### Example implementations
See the adversarial maze environment, [`envs/multigrid/adversarial.py`](https://github.com/facebookresearch/dcd/tree/master/envs/multigrid), for a straightforward example of how these methods and attributes can be implemented.

### Adding policy models
4. Create a new file in [`models`](https://github.com/facebookresearch/dcd/tree/master/models) called `<your_environment>_models.py` containing student and teacher policy models for your environment. See [`models/multigrid_models.py`]([https://github.com/facebookresearch/dcd/tree/master/envs/](https://github.com/facebookresearch/dcd/blob/main/models/multigrid_models.py)) for an example implementation.

### Connecting your environment to the training loop
5. In `util/__init__.py`, update `_make_env` to properly initialize a non-vectorized version of your adversarial environment (e.g. apply any non-vectorized wrappers).
6. In `util/__init__.py`, update `create_parallel_env` to properly initialize a vectorized version of your adversarial environment (e.g. apply any vectorized wrappers).
7. In `util/__init__.py`, update `is_dense_reward_env` to return `True` if your environment returns non-zero rewards before the terminal step.
8. In `util/make_agent.py`, create a new function called `make_model_for_<your_environment>_agent`, following the pattern in the equivalent function for the existing environments, and integrate the call to this function inside the function `make_agent`.
9. In `envs/runners/adversarial_runner.py`, update `use_byte_encoding` to return `True` if the level data structures input into `reset_to_level(level)` for your environment are  Numpy arrays, which will then be turned into byte strings when stored in the PLR level replay buffer.
10. In `envs/runners/adversarial_runner.py`, add support for your environment in `_get_active_levels` by appropriately returning the proper level data structure for your environment's levels. This should be the same data structure that your environment's `reset_to_level(level)` expects as input.
11. You will likely need to define a custom attribute wrapper inside `ParallelAdversarialVecEnv` to return a list of specific per-environment attributes. See the method `get_num_blocks` for an example. In the future, we plan to simplify this implementation by returning all reported level metrics in a single method call. 
12. In `envs/runners/adversarial_runner.py`, update `_get_env_stats` to return a dictionary of mean level metrics and stats across the current batch of vectorized environments. A simple way to do this is to define a function `_get_env_stats_<your_environment>` that returns this dictionary
    
### Connecting your environment to the evaluation logic
In order to perform zero-shot transfer evaluation of agents trained in your adversarial environment, you should create customized, out-of-distribution versions your environment, typically done by subclassing the environment (same base class as used by `<YourEnv>Adversarial`).    

13. In `eval.py`, update `_make_env` in the `Evaluator` class to properly initialize a non-vectorized version of your environment (typically subclass of your environment for zero-shot transfer).
14. In `eval.py`, update `wrap_venv` in the `Evaluator` class to properly initialize a vectorized version of the environment created in (12).
15. If your environment integration runs well and could be of interest to many researchers, consider making a pull request to integrate the environment into this repo.