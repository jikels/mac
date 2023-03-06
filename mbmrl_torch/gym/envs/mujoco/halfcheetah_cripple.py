from mbmrl_torch.gym.envs.mujoco.halfcheetah_base import HalfCheetahBase
from mbmrl_torch.gym.envs.meta_env import MetaEnv
from mbmrl_torch.gym.utils.helper_functions import get_action_dim
import numpy as np
import os

class HalfCheetahCrippleEnv(HalfCheetahBase,MetaEnv):
    
    def __init__(self, task={'crippled_joint': None}):
        # init env
        xml_path = os.path.abspath('mbmrl_torch/gym/envs/mujoco/assets/half_cheetah.xml')
        self.init_cripple_mask = np.ones(6)
        self.cripple_mask = self.init_cripple_mask
        super(HalfCheetahCrippleEnv, self).__init__(xml_path)

        # init task
        self.init_rgba = self.model.geom_rgba.copy()
        self.set_task(task)

    # -------- MetaEnv Methods --------
    def sample_tasks(self, num_tasks):
        joints = np.random.randint(0, 6, size=num_tasks)
        tasks = [{'crippled_joint': joint} for joint in joints]
        return tasks

    def set_task(self, task):
        self.cripple_mask = self.init_cripple_mask
        self.model.geom_rgba[:] = self.init_rgba
        self.task = task
        crippled_joint = task['crippled_joint']
        if crippled_joint is not None:
            self.cripple_mask[crippled_joint] = 0
            geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint+3])
            self.model.geom_rgba[geom_idx, :3] = np.array([1, 0, 0])

    # -------- Gym Methods --------
    # adjusted step function
    def step(self, action):
        # apply cripple mask 
        action = self.cripple_mask * action
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost)
        return (observation, reward, done, infos)
