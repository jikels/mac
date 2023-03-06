# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ..runners import base
from scipy.spatial.distance import cosine as similarity
import cherry as ch
import wandb

class Runner(base.RunnerBase):
    def __init__(
        self,
        config,
        configs_training_tasks,
        configs_testing_tasks):

        print("Initializing task environment runner...")
        super(Runner, self).__init__(
            config=config,
            configs_training_tasks=configs_training_tasks,
            configs_testing_tasks=configs_testing_tasks,
        )

    def run_episode(self):
        self.reward = 0
        reward_per_episode = 0
        self.ExperienceReplay = ch.ExperienceReplay()
        
        for _ in range(self.config["episode_steps"]):

            action, expected_state = self.get_action(self.state)

            #take step in environment and add it to experience replay buffer
            next_state, self.reward, done, _ = self.env.step(action)
            self.ExperienceReplay.append(self.state, action, self.reward, next_state, done)
            self.total_reward += self.reward
            reward_per_episode = reward_per_episode + self.reward

            if self.config["record_video"]==True:
                self.recorder.capture()

            self.state = next_state
            
            # only adapt model if meta testing
            try:
                if self.config["train_online"]:
                    self.adapt_model()
                else:
                    pass
            except:
                self.adapt_model()

            #Experiment tracking via Weights and Biases
            wandb.log({
                'reward_per_step': self.reward,
                'cumulated_reward_per_episode':reward_per_episode,
                'total_reward':self.total_reward,
                'states':wandb.Histogram(self.state)}, step=self.step)

            try:
                wandb.log({
                    'similarity_state_expected_real':similarity(self.state,expected_state)}, step=self.step)
            except:
                pass
                
            self.step=self.step+1
