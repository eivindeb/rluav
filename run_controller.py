import stable_baselines.common
import os.path as osp
import utils
import json
import numpy as np
from stable_baselines.common.vec_env import VecNormalize, VecEnv
from stable_baselines.her import HERGoalEnvWrapper


if __name__ == "__main__":
    env_config_kw = None  #e.g. for shorter episode with a single target: {"target": {"resample_every": 0}, "steps_max": 150}
    sim_config_kw = None

    env, model_folder, _, _, seed, model_type = utils.create_env(monitor=False, allow_new=False, env_config_kw=env_config_kw, sim_config_kw=sim_config_kw, env_attrs={"delay_until_reset": False})
    if model_type is None:
        model_type = "best"

    if isinstance(env, VecNormalize) or (isinstance(env, HERGoalEnvWrapper) and env.norm) or (isinstance(env, VecEnv) and env.get_attr("norm")[0]):
        env.load_running_average(osp.join(model_folder, model_type))
        env.training = False

    env.seed(seed)

    with open(osp.join(model_folder, "rl_config.json")) as config_file:
        rl_config = json.load(config_file)

    algorithm = getattr(stable_baselines, rl_config["algorithm"])

    model_file_path = osp.join(model_folder, model_type, "model.zip")

    model = algorithm.load(load_path=model_file_path, env=env)

    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
    env.render(mode="plot")


