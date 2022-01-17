import json
import stable_baselines
from stable_baselines.common.vec_env import VecNormalize, VecEnv, VecEnvWrapper
import os.path as osp
import os
import numpy as np
import tqdm
import time
import utils
#from stable_baselines.gail import ExpertDataset
import tensorflow as tf
import tensorboard.program
from tensorboard.util import tb_logging
import logging

from collections import deque

from utils import get_attr


def save_model(model, type, step=None, save_replay_buffer=False):
    global checkpoint_folder, best_folder, model_folder, rl_config
    if type == "checkpoint":
        path = checkpoint_folder
    elif type == "best":
        path = best_folder
    elif type == "test":
        path = osp.join(model_folder, "test")
    else:
        raise ValueError
    if step is not None:
        step = round(step, -3)

    model_env = model.get_env()
    if isinstance(model_env, VecNormalize) or rl_config.get("her_norm", False):
        model_env.save_running_average(path, suffix=str(step) if step is not None else None)

    path = osp.join(path, "model")
    if step is not None:
        path += "_{}".format(step)
    model.save(path, save_replay_buffer=save_replay_buffer)


def monitor_training(_locals, _globals):
    global pbar, env, rl_config, model_folder, checkpoint_folder, best_folder, stats, last_save, info_kw, last_ep_info, info
    pbar.total = _locals["total_timesteps"]
    pbar.update(_locals["self"].num_timesteps - pbar.n)
    now = time.time()

    if len(_locals["self"].ep_info_buf) > 0:
        if last_ep_info != _locals["self"].ep_info_buf[-1]:
            last_ep_info = _locals["self"].ep_info_buf[-1]

            stats["num_episodes"] += 1
            ep_rews = [ep_info['r'] for ep_info in _locals["self"].ep_info_buf]
            mean_ep_rew = np.nan if len(ep_rews) == 0 else np.mean(ep_rews)
            stats["last_mean_reward"] = mean_ep_rew
            stats["ep_info_buf"] = list(_locals["self"].ep_info_buf)

            if mean_ep_rew > stats["best_mean_reward"] or np.isnan(stats["best_mean_reward"]):
                save_model(_locals["self"], "best")
                stats["best_mean_reward"] = mean_ep_rew

            info = {}
            for ep_info in _locals["self"].ep_info_buf:
                for k in ep_info.keys():
                    if k in info_kw:
                        if k not in info:
                            info[k] = {}
                        if isinstance(ep_info[k], dict):
                            for state, v in ep_info[k].items():
                                if state in info[k]:
                                    info[k][state].append(v)
                                else:
                                    info[k][state] = [v]
                        else:
                            if "all" in info[k]:
                                info[k]["all"].append(ep_info[k])
                            else:
                                info[k]["all"] = [ep_info[k]]

            if _locals["writer"] is not None:
                _locals["writer"].add_summary(tf.Summary(value=[tf.Summary.Value(tag="ep_info/num_episodes", simple_value=stats["num_episodes"])]), _locals["self"].num_timesteps)
                if "success" in info:
                    summaries = []
                    for measure in info_kw:
                        for k, v in info[measure].items():
                            summaries.append(tf.Summary.Value(tag="ep_info/{}_{}".format(measure, k), simple_value=np.nanmean(v)))
                    _locals["writer"].add_summary(tf.Summary(value=summaries), _locals["self"].num_timesteps)

            elif _locals["update"] % rl_config["log_interval"] == 0 and _locals["update"] != 0:
                for info_k, info_v in info.items():
                    tqdm.tqdm.write("\n{}:\n\t".format(info_k) + "\n\t".join(["{:<10s}{:.2f}".format(k, np.nanmean(v)) for k, v in info_v.items()]))

    elif _locals["self"].ep_info_buf.maxlen > 10:
        _locals["self"].ep_info_buf = deque(maxlen=10)

    elif len(stats["ep_info_buf"]) > len(_locals["self"].ep_info_buf):
        _locals["self"].ep_info_buf = deque(stats["ep_info_buf"])
        last_ep_info = stats["ep_info_buf"][-1]

    if now - last_save >= rl_config["save_interval"]:
        last_save = time.time()
        save_model(_locals["self"], "checkpoint")

    with open(osp.join(model_folder, "stats.json"), "w") as stats_file:
        json.dump(stats, stats_file)

    return True


if __name__ == "__main__":
    last_ep_info = None
    info = None

    env, model_folder, load, tb_info, _, _ = utils.create_env(monitor=True, allow_new=True)
    checkpoint_folder = osp.join(model_folder, "checkpoint")
    best_folder = osp.join(model_folder, "best")

    env_cfg = get_attr(env, "cfg")[0] if isinstance(env, VecEnv) or isinstance(env, VecEnvWrapper) else get_attr(env, "cfg")[0]

    info_kw = [metric["name"] for metric in env_cfg.get("metrics", [])] + ["reward"]

    with open(osp.join(model_folder, "rl_config.json")) as config_file:
        rl_config = json.load(config_file)

    if osp.exists(osp.join(model_folder, "stats.json")):
        with open(osp.join(model_folder, "stats.json")) as stats_file:
            stats = json.load(stats_file)
    else:
        stats = {"best_mean_reward":  np.nan, "last_mean_reward": np.nan, "last_test": 0, "num_episodes": 0,
                 "ep_info_buf": []}
    obs_shape = env_cfg["observation"]["shape"]

    if rl_config["log_tensorboard"]:
        tensorboard_folder = osp.join(model_folder, "tensorboard")
        rl_config["model_args"]["tensorboard_log"] = tensorboard_folder
        if tb_info["port"] is not None:
            # Remove request prints etc.
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            tf_log = logging.getLogger("tensorflow")
            tf_log.setLevel(logging.ERROR)
            # Start tensorboard
            tb = tensorboard.program.TensorBoard()
            # Need host argument when ipv6 is disabled
            tb.configure(argv=[None, "--host", str(tb_info["ip"]), "--port", str(tb_info["port"]), "--logdir", tensorboard_folder])
            url = tb.launch()
            print("Launched tensorboard at {}".format(url))
            tb_logging.get_logger().setLevel(logging.ERROR)

    alg_kw = dict(
        env=env,
        policy_kwargs=rl_config["policy"].get("kwargs", {})
    )
    alg_kw.update(rl_config["model_args"])

    exclusive_states = get_attr(env, "obs_exclusive_states", indices=0)[0]
    if exclusive_states:
        alg_kw["policy_kwargs"]["obs_module_indices"] = get_attr(env, "obs_module_indices", indices=0)[0]

    algorithm = getattr(stable_baselines, rl_config["algorithm"])

    if rl_config.get("use_her", False):
        alg_kw["model_class"] = algorithm
        alg_kw["norm"] = rl_config.get("her_norm", False)
        alg_kw["n_sampled_goal"] = rl_config.get("n_sampled_goal", 4)
        alg_kw["goal_selection_strategy"] = rl_config.get("goal_selection_strategy", "future")
        if rl_config.get("goal_mode", "absolute") == "relative":
            alg_kw["policy_kwargs"]["goal_size"] = len(get_attr(env, "cfg")[0]["observation"].get("goals", []))
        algorithm = stable_baselines.her.HER

    if load:
        load_path = osp.join(checkpoint_folder, "model")
        model = algorithm.load(load_path=load_path, **alg_kw)
        if isinstance(env, VecNormalize):
            env.load_running_average(checkpoint_folder)
        print("Loaded model {}".format(model_folder))
    else:
        if rl_config["algorithm"] == "SAC":
            policy = getattr(stable_baselines.sac, rl_config["policy"]["name"])
        else:
            policy = getattr(stable_baselines.common.policies, rl_config["policy"]["name"])
        model = algorithm(policy, **alg_kw)

    print("Training model {}".format(model_folder))
    pbar = tqdm.tqdm(desc="Training Agent")
    last_save = time.time()
    last_render = time.time()
    render_check = {"files": [], "time": time.time()}

    model.learn(total_timesteps=int(rl_config["timesteps"]), log_interval=rl_config["log_interval"],
                callback=monitor_training, reset_num_timesteps=not load)
    save_model(model, "checkpoint", save_replay_buffer=rl_config.get("gather_data", False))
    exit(0)
