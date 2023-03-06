from mbmrl_torch.gym.envs.mujoco.ant_base import AntBaseEnv
from mbmrl_torch.gym.envs.meta_env import MetaEnv
import numpy as np

class AntCrippleEnv(AntBaseEnv, MetaEnv):

    """
    **Credit**

    Code adapted from https://github.com/iclavera/learning_to_adapt/blob/bd7d99ba402521c96631e7d09714128f549db0f1/learning_to_adapt/envs/ant_env.py#L2

    """

    def __init__(self, task={'crippled_leg':0}):
        # env init
        super(AntCrippleEnv, self).__init__()
        
        # task init
        self.geom_names = self.model.geom_names
        self.body_names = self.model.body_names

        self.init_geom_sizes = self.model.geom_size.copy()
        self.init_body_ipos = self.model.body_ipos.copy()
        self.init_geom_rgba = self.model.geom_rgba.copy()

        self.set_task(task)

    # -------- MetaEnv Methods --------

    def _init_task(self):
        # reset color
        for i in range(len(self.init_geom_rgba)):
            self.model.geom_rgba[i] = self.init_geom_rgba[i]
        # reset sizes
        for i in range(len(self.init_geom_sizes)):
            self.model.geom_size[i] = self.init_geom_sizes[i]
        # reset positions
        for i in range(len(self.init_body_ipos)):
            self.model.body_ipos[i] = self.init_body_ipos[i]

    def sample_tasks(self, num_tasks):
        # sample the leg to remove (1-3 train / 4 test)
        legs = []
        for _ in range(num_tasks):
            legs.append(np.random.randint(1, 4))

        tasks = [{'crippled_leg':legs[i]} for i in range(len(legs))]

        return tasks

    def set_task(self, task):
        self._init_task()
        self.task = task
        crippled_leg = task['crippled_leg']

        if crippled_leg is not None:
            
            # define relevant leg ids
            leg_id_geom = self.model.geom_name2id(
                self.geom_names[3*crippled_leg])
            
            leg_id_body = self.model.body_name2id(
                self.body_names[2*crippled_leg])

            # Make the removed leg look red
            self.model.geom_rgba[leg_id_geom] = np.array([1, 0, 0, 1])
            self.model.geom_rgba[leg_id_geom+1] = np.array([1, 0, 0, 1])

            # Make the removed leg not affect anything
            temp_size_top = self.init_geom_sizes[leg_id_geom]/2
            temp_size_bottom = self.init_geom_sizes[leg_id_geom+1]/2

            # local position of center of mass of bottom
            temp_pos_bottom = self.init_body_ipos[leg_id_body+1]
            
            # alter the mujoco model
            self.model.geom_size[leg_id_geom] = temp_size_top
            self.model.geom_size[leg_id_geom+1] = temp_size_bottom
            self.model.body_ipos[leg_id_geom+1] = temp_pos_bottom