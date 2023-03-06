import numpy as np
import torch
from gym.error import DependencyNotInstalled

try:
    from gym.envs.mujoco.mujoco_env import MujocoEnv
except DependencyNotInstalled:
    from mbmrl_torch.gym.envs.mujoco.dummy_mujoco_env import MujocoEnv


class AntBaseEnv(MujocoEnv):
    
    """
    **Credit**

    Code adapted from https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/mujoco/ant_direction.py

    """

    def __init__(self):
        self.goal_direction = [1.0, 0.0]
        super(AntBaseEnv, self).__init__("ant.xml", 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Mujoco Methods --------
    def _get_obs(self):
        return np.concatenate(
            [
                # self.get_body_com("torso").flat, #0:xpos 1:ypos 2:zpos of
                # torso -> same as below without quaternion
                self.sim.data.qpos.flat,
                # first 7 elements: 3D position (x,y,z) and orientation
                # (quaternion x,y,z,w) of the torso /remaining 8 elements: the
                # joint angles
                self.sim.data.qvel.flat,  # angular velocity of all 8 joints
                # self.sim.data.get_body_xmat("torso").flat, #frame orientation -> not relevant
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat, #contact costs -> not relevant
            ]
        )

    def set_obs(self, obs):
        qpos = obs[0:15]  # extract qpos
        qvel = obs[15:]  # extrac qval
        self.set_state(qpos, qvel)  # set qpos an qval

    def viewer_setup(self):
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        # Hide the overlay
        self.viewer._hide_overlay = True

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

    # -------- Gym Methods --------
    def step(self, action):
        # get observation t0 and read positions
        obs_before = self._get_obs()
        xy_posbefore = obs_before[:2]

        # step
        self.do_simulation(action, self.frame_skip)

        # get observation t1 and read positions
        ob = self._get_obs()
        xy_posafter = ob[:2]

        forward_reward = (
            np.sum(np.array(self.goal_direction) * (xy_posafter - xy_posbefore)) / self.dt
        )

        ctrl_cost = 0.005 * np.square(action).sum()

        survive_reward = 0.05

        notdone = np.isfinite(ob[2]).all() and ob[2] >= 0.2 and ob[2] <= 1.0
        done = not notdone

        reward = forward_reward + survive_reward - ctrl_cost

        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                cost_control=-ctrl_cost,
                done=done,
                reward_survive=survive_reward,
            ),
        )

    # get reward by using predicted states from dynamics model during MPC /
    # todo: add contact cost
    def get_reward_array(self, actions, start_states, next_states):
        xy_posbefore = start_states[:, :2]  # [popsize:observation_len]
        xy_posafter = next_states[:, :2]  # [popsize:observation_len]

        direction_array = np.full(
            (xy_posbefore.shape), self.goal_direction
        )  # make array of position with as much elements as there are states
        forward_rewards = (
            np.sum(direction_array * (xy_posafter - xy_posbefore), axis=1) / 0.05
        )  # calculating reward

        actions = 0.005 * np.square(actions)  # square actions
        ctrl_costs = np.sum(actions, axis=1)
        survive_rewards = np.full((forward_rewards.shape), 0.05)

        reward = forward_rewards + survive_rewards - ctrl_costs
        # reward = np.where(np.logical_and(next_states[:,2]<=1.0, next_states[:,2]>=0.5),forward_rewards + survive_rewards - ctrl_costs,- ctrl_costs)
        return reward

    # get reward by using predicted states from dynamics model during MPC
    def get_reward_tensor(
        self,
        actions: torch.Tensor,
        start_states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        xy_posbefore = start_states[:, :2]  # [popsize:observation_len]
        xy_posafter = next_states[:, :2]  # [popsize:observation_len]

        direction_tensor = torch.tile(
            input=torch.tensor(self.goal_direction), dims=(xy_posafter.size(axis=0), 1)
        ).to(self.device)
        forward_rewards = (
            torch.sum(direction_tensor * (xy_posafter - xy_posbefore), axis=1) / 0.05
        )  # calculating reward

        actions = 0.005 * torch.square(actions)  # square actions
        ctrl_costs = torch.sum(actions, axis=1)
        survive_rewards = torch.full_like(forward_rewards, 0.05)

        reward = forward_rewards + survive_rewards - ctrl_costs
        # reward = np.where(np.logical_and(next_states[:,2]<=1.0, next_states[:,2]>=0.5),forward_rewards + survive_rewards - ctrl_costs,- ctrl_costs)
        return reward

    def reset(self, *args, **kwargs):
        MujocoEnv.reset(self, *args, **kwargs)
        return self._get_obs()
