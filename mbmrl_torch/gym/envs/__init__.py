#!/usr/bin/env python3

from gym.envs.registration import register


# 2D Navigation
# ----------------------------------------

register(
    'Particles2D-v1',
    entry_point='mbmrl_torch.gym.envs.particles.particles_2d:Particles2DEnv',
    max_episode_steps=100,
)

# Mujoco
# ----------------------------------------

register(
    'HalfCheetahBase-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.halfcheetah_base:HalfCheetahBase',
    max_episode_steps=1000,
)

register(
    'HalfCheetahForwardBackward-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.halfcheetah_forward_backward:HalfCheetahForwardBackwardEnv',
    max_episode_steps=1000,
)

register(
    'HalfCheetahBlocks-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.halfcheetah_blocks:HalfCheetahBlocksEnv',
    max_episode_steps=1000,
)

register(
    'HalfCheetahHField-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.halfcheetah_hfield:HalfCheetahHFieldEnv',
    max_episode_steps=1000,
)

register(
    'HalfCheetahCripple-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.halfcheetah_cripple:HalfCheetahCrippleEnv',
    max_episode_steps=1000,
)

register(
    'AntDirection-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.ant_direction:AntDirectionEnv',
    max_episode_steps=1000,
)

register(
    'AntGravity-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.ant_gravity:AntGravityEnv',
    max_episode_steps=1000,
)

register(
    'AntCripple-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.ant_cripple:AntCrippleEnv',
    max_episode_steps=1000,
)

register(
    'HumanoidForwardBackward-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.humanoid_forward_backward:HumanoidForwardBackwardEnv',
    max_episode_steps=1000,
)

register(
    'HumanoidDirection-v1',
    entry_point='mbmrl_torch.gym.envs.mujoco.humanoid_direction:HumanoidDirectionEnv',
    max_episode_steps=1000,
)

register(
    'FetchReachMeta-v1',
    entry_point='mbmrl_torch.gym.envs.robotics.reach:FetchReachEnv',
    max_episode_steps=50,
)

register(
    'FetchPickAndPlaceMeta-v1',
    entry_point='mbmrl_torch.gym.envs.robotics.pick_and_place:FetchPickAndPlaceEnv',
    max_episode_steps=50,
)

register(
    'FetchPushMeta-v1',
    entry_point='mbmrl_torch.gym.envs.robotics.push:FetchPushEnv',
    max_episode_steps=50,
)


