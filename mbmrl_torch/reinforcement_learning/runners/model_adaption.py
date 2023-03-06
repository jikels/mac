# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time 
import wandb
from mbmrl_torch.utils.data_processing import process_experience_replay

from .env_runner import Runner

class OnlineModelAdaption(Runner):

    def __init__(
        self,
        config,
        configs_training_tasks,
        configs_testing_tasks,
        model_handler
        ):
        print("Initializing model adaptor...")

        super(Runner, self).__init__(
            config=config,
            configs_training_tasks=configs_training_tasks,
            configs_testing_tasks=configs_testing_tasks,
        )
        self.model_handler = model_handler
        self.model_handler.init_online_training()
        self.time_per_model_step = 0
        self.mean_loss_per_step = []
        self.reward_per_m_steps = []
        self.observations = 0

    def adapt_model(self):
        self.observations = self.observations+1
        
        mean_loss = 0
        reward_m_steps = [0]
        if self.observations >= self.config["m_observations"]:
            start_time = time.time()

            #train the model based on the recent m observations
            memory = -1*(self.config["m_observations"])
            x,y,_,_,reward_label = process_experience_replay(self.ExperienceReplay[memory:])

            mean_loss = self.model_handler.train_model_online(train_in=x, train_out=y)
            wandb.log({'mean_loss_per_step': mean_loss}, step=self.step)
            self.mean_loss_per_step.append(mean_loss)
            
            reward_m_steps = sum(self.ExperienceReplay[memory:].reward())
            reward_m_steps= reward_m_steps.numpy()
                
            #remove oldest observation
            self.time_per_model_step = time.time() - start_time

        normalization_statistics = self.model_handler.normalizer.get_stats()

        #Experiment tracking via Weights and Biases
        wandb.log({
            'total_reward_last_'+str(self.config["m_observations"])+'_steps': reward_m_steps[0],
            'time_per_model_step': self.time_per_model_step,
            'normalizer_mean_input': wandb.Histogram(normalization_statistics[0]),
            'normalizer_mean_output': wandb.Histogram(normalization_statistics[1]),
            'normalizer_std_input': wandb.Histogram(normalization_statistics[2]),
            'normalizer_std_output': wandb.Histogram(normalization_statistics[3])}, step=self.step)