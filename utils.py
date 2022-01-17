import gym
import matplotlib.pyplot as plt

from gym_fixed_wing.fixed_wing import FixedWingAircraft, FixedWingAircraftGoal
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from stable_baselines.common import set_global_seeds
    from stable_baselines.td3 import TD3
    from stable_baselines.sac import SAC
    from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnvWrapper, VecEnv
    from stable_baselines.her import HERGoalEnvWrapper
    from stable_baselines.bench import Monitor
import os.path as osp
import os
import shutil
import numpy as np
import json
import pickle


def make_env(env_class, init_kw, rank, seed=0, info_kw=(), monitor=True, training=True, HER=False, her_norm=False, her_norm_rew=False, attrs=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = env_class(**init_kw)
        env.training = training
        env.seed(seed + rank)
        info_kw = [metric["name"] for metric in env.cfg.get("metrics", {})] + ["reward"]
        if attrs is not None:
            for k, v in attrs.items():
                setattr(env, k, v)
        if monitor:
            env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=info_kw)
        if HER:
            env = HERGoalEnvWrapper(env, norm=her_norm, norm_reward=her_norm_rew)
        return env
    set_global_seeds(seed)
    return _init


def get_attr(_env, item, **kwargs):
    if isinstance(_env, VecEnv) or isinstance(_env, VecEnvWrapper):
        return _env.get_attr(item, **kwargs)
    #elif isinstance(_env, HERGoalEnvWrapper):
    #    return [getattr(_env.env, item)]
    else:
        return [getattr(_env, item)]


def env_get_attr(_env, item, her_wrapper=False, **kwargs):
    if isinstance(_env, VecEnv) or isinstance(_env, VecEnvWrapper):
        if her_wrapper:
            return _env.env_method("env_get_attr", (item))
        else:
            return _env.get_attr(item, **kwargs)
    elif her_wrapper:
        return [_env.env_get_attr(item)]
    else:
        return [getattr(_env, item)]


def env_set_attr(_env, item, val, her_wrapper=False, **kwargs):
    if isinstance(_env, VecEnv) or isinstance(_env, VecEnvWrapper):
        if her_wrapper:
            return _env.env_method("env_set_attr", (item, val))
        else:
            return _env.set_attr(item, val, **kwargs)
    elif her_wrapper:
        return [_env.env_set_attr(item, val)]
    else:
        return [setattr(_env, item, val)]


def linear_transformation(val, new_max, new_min, old_max, old_min):
    return np.array(new_max - new_min) * (val - old_min) / (old_max - old_min) + new_min


def create_env(monitor, seed=None, allow_new=True, env_config_kw=None, sim_config_kw=None, args=None, n_proc=None, reset_on_done=True, env_attrs=None, norm=True):
    from argparse import ArgumentParser

    if args is None:
        parser = ArgumentParser()
        parser.add_argument("model_folder", help="Path to model folder. If already exists, configurations will be loaded from this folder and training will resume from checkpoint.")
        parser.add_argument("--rl_config_path", required=False, help="Path to configuration for RL algorithm")
        parser.add_argument("--env_config_path", required=False, help="Path to configuration for gym environment")
        parser.add_argument("--sim_config_path", required=False, help="Path to configuration for PyFly simulator")
        parser.add_argument("--aircraft_param_path", required=False, help="Path to parameter file for aircraft")
        parser.add_argument("--training", dest="training", action="store_true", help="Set training attribute of environment")
        parser.add_argument("--tensorboard_ip", required=False, help="IP to launch tensorboard server on", default=None)
        parser.add_argument("--tensorboard_port", required=False, help="Port to launch tensorboard server on", default=None)
        parser.add_argument("--which_model", required=False, help="Which model (checkpoint/best) to load", default=None)
        parser.add_argument("--seed", required=False, help="Seed", default=0)
        parser.set_defaults(training=False)

        args = parser.parse_args()
    if n_proc is not None:
        args.n_proc = n_proc
    else:
        args.n_proc = 1

    load = False
    model_folder = args.model_folder
    checkpoint_folder = osp.join(model_folder, "checkpoint")
    best_folder = osp.join(model_folder, "best")

    if osp.exists(checkpoint_folder) and (len(os.listdir(checkpoint_folder)) > 0 or not allow_new):
        load = True
    else:
        if osp.exists(model_folder):
            shutil.rmtree(model_folder)
        if allow_new:
            try:
                rl_config_path = args.rl_config_path
                env_config_path = args.env_config_path
                sim_config_path = args.sim_config_path
                aircraft_param_path = args.aircraft_param_path

                os.makedirs(model_folder)
                os.makedirs(checkpoint_folder)
                os.makedirs(best_folder)
                os.makedirs(osp.join(model_folder, "tensorboard"))
                os.makedirs(osp.join(model_folder, "render"))

                shutil.copy2(rl_config_path, osp.join(model_folder, "rl_config.json"))
                shutil.copy2(env_config_path, osp.join(model_folder, "env_config.json"))
                shutil.copy2(sim_config_path, osp.join(model_folder, "sim_config.json"))
                shutil.copy2(aircraft_param_path, osp.join(model_folder, "aircraft_params.mat"))
            except Exception as e:
                print("All configuration files must be provided as arguments when creating new model.")
                raise e
        else:
            raise Exception("Model folder must already exist when keyword allow_new is False")

    rl_config_path = osp.join(model_folder, "rl_config.json")
    env_config_path = osp.join(model_folder, "env_config.json")
    sim_config_path = osp.join(model_folder, "sim_config.json")
    aircraft_param_path = osp.join(model_folder, "aircraft_params.mat")

    with open(rl_config_path) as config_file:
        rl_config = json.load(config_file)

    if seed is None:
        seed = rl_config.get("seed", args.seed)

    if not load and rl_config.get("offline_data_path", None) is not None:
        shutil.copy2(os.path.join(os.path.dirname(rl_config["offline_data_path"]), "obs_rms.pkl"),
                     os.path.join(checkpoint_folder, "obs_rms.pkl"))
        shutil.copy2(os.path.join(os.path.dirname(rl_config["offline_data_path"]), "ret_rms.pkl"),
                     os.path.join(checkpoint_folder, "ret_rms.pkl"))

    env_init = {"config_path": env_config_path, "sim_config_path": sim_config_path,
                "sim_parameter_path": aircraft_param_path, "config_kw": env_config_kw, "sim_config_kw": sim_config_kw}

    if rl_config.get("use_her", False):
        env_class = FixedWingAircraftGoal
        reset_on_done = False
    else:
        env_class = FixedWingAircraft

    if rl_config["algorithm"] in ["PPO"] or int(args.n_proc) > 1:
        if int(args.n_proc) > 1:
            env = SubprocVecEnv([make_env(env_class, env_init, i, seed=seed, monitor=monitor, training=args.training, HER=rl_config.get("use_her", False), her_norm=rl_config.get("her_norm", False) and norm, her_norm_rew=rl_config.get("her_norm_rew", False), attrs=env_attrs) for i in range(int(args.n_proc))], reset_on_done=reset_on_done)
        else:
            env = DummyVecEnv([make_env(env_class, env_init, 0, seed=seed, monitor=monitor, training=args.training, HER=rl_config.get("use_her", False), her_norm=rl_config.get("her_norm", False) and norm, her_norm_rew=rl_config.get("her_norm_rew", False), attrs=env_attrs)])
    else:
        env = make_env(env_class, env_init, 0, seed=seed, monitor=monitor, training=args.training, HER=rl_config.get("use_her", False), her_norm=rl_config.get("her_norm", True) and norm, her_norm_rew=rl_config.get("her_norm_rew", False), attrs=env_attrs)()

    if rl_config.get("vec_normalize", False):
        env = VecNormalize(env, gamma=rl_config.get("model_params", {}).get("gamma", 0.99), mean_mask=env.get_attr("obs_norm_mean_mask")[0])

    tensorboard_info = {"ip": args.tensorboard_ip, "port": args.tensorboard_port}

    return env, model_folder, load, tensorboard_info, seed, args.which_model
