from gym.spaces import Box, Discrete, Dict, Tuple

def is_vectorized(env):
    return hasattr(env, 'num_envs') and env.num_envs > 1

def is_vector_env(env):
    return getattr(env, "is_vector_env", False)

def get_observation_dim(env):
    space = env.observation_space
    o = get_dim(space)
    if is_vectorized(env):
        o = o // env.num_envs
    return o

def get_action_dim(env):

    if is_vectorized(env):
        space = env.action_space[0]
    else:
        space = env.action_space
        
    a = get_dim(space)
    return a

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))

def sample_tasks_VecEnv(env, num_tasks):
    tasks = env.unwrapped.sample_tasks(num_tasks)
    return tasks