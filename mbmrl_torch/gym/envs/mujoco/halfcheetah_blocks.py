from mbmrl_torch.gym.envs.mujoco.halfcheetah_base import HalfCheetahBase
from mbmrl_torch.gym.envs.meta_env import MetaEnv
import numpy as np
import os

class HalfCheetahBlocksEnv(HalfCheetahBase,MetaEnv):
    
    def __init__(self, task={'damping': np.array([6.0,4.5,3.0,4.5,3.0,1.5])}):
        # init env
        xml_path = os.path.abspath('mbmrl_torch/gym/envs/mujoco/assets/half_cheetah_blocks.xml')
        super(HalfCheetahBlocksEnv, self).__init__(xml_path)
        
        # init task
        self.init_damping = self.model.dof_damping.copy()
        self.set_task(task)
    
    # -------- Mujoco Methods --------
    # adjusted obs get the correct state dim
    # and state information
    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[8:],
                self.sim.data.qvel.flat[8:],
            ]
        )

    # -------- Gym Methods --------
    # adjusted step function to extract
    # the correct state information
    def step(self, action):
        xposbefore = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[8]
        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost)
        return (observation, reward, done, infos)

    # -------- MetaEnv Methods --------
    def sample_tasks(self, num_tasks):
        dampings = np.random.uniform(2, 10, size=(num_tasks, 6))
        dampings = np.round(dampings,1)
        tasks = [{'damping': damping} for damping in dampings]
        return tasks

    def set_task(self, task):
        self.model.dof_damping[:] = self.init_damping
        self.task = task
        damping = task['damping']
        self.model.dof_damping[11:] = damping