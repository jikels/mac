# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cherry as ch
import numpy as np
import os
import random
from tqdm import tqdm
from mbmrl_torch.gym.utils.env_init import init_env
from mbmrl_torch.gym.utils.helper_functions import get_action_dim, get_observation_dim, get_dim

def generate_train_test_tasks(env, n_tasks, meta_train_test_split):
    task_configs = env.sample_tasks(n_tasks)
    configs_training_tasks = task_configs[
        : round(len(task_configs) * meta_train_test_split, None)
    ]
    configs_testing_tasks = task_configs[
        round(len(task_configs) * meta_train_test_split, None) :
    ]
    return configs_training_tasks, configs_testing_tasks

def generate_training_data(
    rollouts,
    episode_length,
    env,
    configs_training_tasks,
    path,
    done_reset=True,
    policy=None,
):
    ae = action_executer(env, policy)
    i = 0
    for task_config in tqdm(configs_training_tasks, leave=False, desc="Data"):
        
        obs = env.reset() # reset env before sampling a new task
        env.set_task(task_config)  # Samples a new config

        ExperienceReplay = ae.execute(
            episode_length=episode_length, rollouts=rollouts, done_reset=done_reset
        )
        a = str(i)

        if os.path.exists(path):
            save_path = path + "/Task_" + a + ".pt"
            ExperienceReplay.save(save_path)
        else:
            os.makedirs(path)
            save_path = path + "/Task_" + a + ".pt"
            ExperienceReplay.save(save_path)

        print("All trajectories saved")
        i = i + 1

# action executer creating experience replay
class action_executer:
    def __init__(self, env, policy=None):
        self.policy = policy
        self.env = env
        self.dict_obs = False

        obs = self.env.reset()
        if type(obs) is dict:
            print("env state is dictionary")
            self.dict_obs = True
        else:
            print("env state is not a dictionary")

    def action(self, state):
        if self.policy == None:
            return self.env.action_space.sample()
        else:
            action, _ = self.policy.predict(state, deterministic=True)
            return np.squeeze(action)

    def execute(self, rollouts, episode_length, done_reset=True):
        ExperienceReplay = ch.ExperienceReplay()  # Manage transitions
        self.env.reset()
        for j in range(rollouts):
            #print("rollout " + str(j))
            state = self.env.reset()
            i = 1
            while i in range(episode_length):
                action = self.action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Build the ExperienceReplay
                if self.dict_obs == False:
                    ExperienceReplay.append(state, action, reward, next_state, done)
                else:
                    _state = state["observation"]
                    _next_state = next_state["observation"]
                    ExperienceReplay.append(_state, action, reward, _next_state, done)

                state = next_state

                if done:
                    if done_reset == True:
                        state = self.env.reset()

                i = i + 1
        return ExperienceReplay

