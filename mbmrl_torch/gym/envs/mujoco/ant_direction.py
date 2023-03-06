from mbmrl_torch.gym.envs.mujoco.ant_base import AntBaseEnv
from mbmrl_torch.gym.envs.meta_env import MetaEnv
import numpy as np

class AntDirectionEnv(AntBaseEnv,MetaEnv):

    def __init__(self, task={'direction': [1.0, 0.0]}):
        # env init
        super(AntDirectionEnv, self).__init__()

        # task init
        self.set_task(task)

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        self.task = task
        self.goal_direction = task['direction']

    def sample_tasks(self, num_tasks):
        directions = np.random.normal(size=(num_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        tasks = [{'direction': [direction[0],direction[1]]} for direction in directions]
        return tasks