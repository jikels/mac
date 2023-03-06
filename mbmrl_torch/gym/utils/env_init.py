from mbmrl_torch.gym.utils.observation_statistics import TrackObservationStats, TrackRewStats
from mbmrl_torch.gym.utils.normalize_env import NormalizeEnv
from mbmrl_torch.gym.utils.action_space_scaler import ActionSpaceScaler
from mbmrl_torch.gym.utils.make import make as make_vec_env
from gym.envs import make as make_single_env

def init_env(
    env_name,
    task=None,
    num_envs=1,
    asynchronous=True,
    seed=42,
    normalized=False,
):
    if num_envs > 1:
        env = make_vec_env(
            env_name,
            num_envs,
            asynchronous,
            task)
        if normalized:
            env = NormalizeEnv(env)
        else:
            env = TrackObservationStats(env)
        env = TrackRewStats(env)
        env = ActionSpaceScaler(env)
    else:
        if task == None:
            env = make_single_env(id=env_name)
        else:
            env = make_single_env(id=env_name, task=task)
        
        if normalized:
            env = NormalizeEnv(env)
        else:
            env = TrackObservationStats(env)
        env = TrackRewStats(env)
        env = ActionSpaceScaler(env)

    # set seed
    env.seed(seed)

    return env
