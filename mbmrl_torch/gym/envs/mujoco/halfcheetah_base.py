import numpy as np
import torch

from gym.error import DependencyNotInstalled

try:
    from gym.envs.mujoco.mujoco_env import MujocoEnv
except DependencyNotInstalled:
    from mbmrl_torch.gym.envs.mujoco.dummy_mujoco_env import MujocoEnv


class HalfCheetahBase(MujocoEnv):
    """
    Adapted from(https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/mujoco/halfcheetah_forward_backward.py)
    """

    def __init__(self, xml_path="mbmrl_torch/gym/envs/mujoco/assets/half_cheetah.xml"):
        self.goal_direction = 1.0
        super(HalfCheetahBase, self).__init__(xml_path, 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Mujoco Methods --------

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        # Hide the overlay
        self.viewer._hide_overlay = True

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    # -------- Gym Methods --------
    def step(self, action):
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
    
    #get reward by using predicted states from dynamics model during MPC (Joel, 25.10.2021)
    def get_reward_array(self, actions, start_states,next_states):
        xposbefore = start_states[:,0] #array
        xposafter = next_states[:,0] #array
        forward_vel = (xposafter - xposbefore) / 0.05 #calculating reward
        forward_reward = self.goal_direction * forward_vel
        actions = 0.5 * 1e-1 * np.square(actions) #square actions
        ctrl_costs = np.sum(actions, axis=1) #cumulate actions
        reward = forward_reward - ctrl_costs 
        return reward
    
    # get reward by using predicted states from dynamics model during MPC
    def get_reward_tensor(
        self,
        actions: torch.Tensor,
        start_states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        xy_posbefore = start_states[:, :1]  # [popsize:observation_len]
        xy_posafter = next_states[:, :1]  # [popsize:observation_len]

        direction_tensor = torch.tile(input=torch.tensor(self.goal_direction), dims=(xy_posafter.size(axis=0), 1)).to(self.device)
        forward_rewards = (torch.sum(direction_tensor * (xy_posafter - xy_posbefore), axis=1) / 0.05)  # calculating reward

        actions = 0.5 * 1e-1 * torch.square(actions)  # square actions
        ctrl_costs = torch.sum(actions, axis=1)

        reward = forward_rewards - ctrl_costs
        
        return reward

    def reset(self, *args, **kwargs):
        MujocoEnv.reset(self, *args, **kwargs)
        return self._get_obs()
