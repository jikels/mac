# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mbmrl_torch.gym.envs.mujoco.ant_base import AntBaseEnv
from mbmrl_torch.gym.envs.meta_env import MetaEnv
import numpy as np

class AntGravityEnv(AntBaseEnv,MetaEnv):

    def __init__(self, task={'gravity': -9.81}):
        # env init
        super(AntGravityEnv, self).__init__()

        # task init
        self.gravity = self.model.opt.gravity #get envs gravity settings

        self.set_task(task)

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        self.task = task
        self.gravity[-1]= task['gravity'] #set gravity setting according to value of generated dictionary of sample_tasks

    def sample_tasks(self, num_tasks):
        gravities = np.random.uniform(low=-19.81, high=-0.90, size=(num_tasks,1)) #get random
        gravities = np.round(gravities, 2)
        tasks = [{'gravity': gravity} for gravity in gravities]
        return tasks
