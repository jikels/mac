import gym
import numpy as np
from .helper_functions import is_vectorized

class ActionSpaceScaler(gym.core.Wrapper):

    """
    Scales the action space to be in the range (-clip, clip).

    Adapted from Vitchyr Pong's RLkit:
    https://github.com/vitchyr/rlkit/blob/master/rlkit/envs/wrappers.py#L41
    """

    def __init__(self, env, clip=1.0):
        super(ActionSpaceScaler, self).__init__(env)
        self.env = env
        self.clip = clip
        ub = np.ones(self.env.action_space.shape) * clip
        self.is_vector_env = getattr(env, "is_vector_env", False)

        if self.is_vector_env:
            lst = list(self.action_space)
            for i in range(len(lst)):
                lst[i] = gym.spaces.Box(-1 * ub, ub, shape=lst[i].shape, dtype=np.float32)
            tpl = tuple(lst)
            self.action_space = tpl
        else:
            self.action_space = gym.spaces.Box(-1 * ub, ub, shape=self.action_space.shape, dtype=np.float32)


    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def _normalize(self, action):
        if self.is_vector_env:
            lb = self.env.action_space[0].low
            ub = self.env.action_space[0].high
        else:
            lb = self.env.action_space.low
            ub = self.env.action_space.high
        
        scaled_action = lb + (action + self.clip) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

    def step(self, action):
        if self.is_vector_env:
            action = [self._normalize(a) for a in action]
        else:
            action = self._normalize(action)
        return self.env.step(action)
