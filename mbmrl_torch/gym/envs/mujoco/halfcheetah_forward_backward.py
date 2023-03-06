from mbmrl_torch.gym.envs.mujoco.halfcheetah_base import HalfCheetahBase
from mbmrl_torch.gym.envs.meta_env import MetaEnv
import numpy as np

class HalfCheetahForwardBackwardEnv(HalfCheetahBase,MetaEnv):
    
    def __init__(self, task={'direction': 1.0}):
        super(HalfCheetahForwardBackwardEnv, self).__init__(xml_path="mbmrl_torch/gym/envs/mujoco/assets/half_cheetah.xml")
        self.set_task(task)
    
    # -------- MetaEnv Methods --------
    def set_task(self, task=1.0):
        self.task = task
        self.goal_direction = task['direction']

    def sample_tasks(self, num_tasks):
        directions = np.random.choice((-1.0, 1.0), (num_tasks,))
        tasks = [{'direction': direction} for direction in directions]
        return tasks
