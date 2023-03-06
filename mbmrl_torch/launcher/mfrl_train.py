# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import wandb
import numpy as np

from mbmrl_torch.reinforcement_learning.algorithms.sac.sac import Sac
from ..neural_networks.model_handler.handler_actor_critic import ModelHandlerAC

class MFRLTraining:
    '''Class for model free reinforcement learning training'''
    def train(self):
        '''Run constructed rl training with wandb'''
        track_run = wandb.init(
            project=self.config["env_name"],
            entity=self.config["wandb_entity"],
            reinit=True,
            config=self.wandb_config,
            job_type=self.wandb_job_type_tr,
            name=self.config["exp_name"],
            group=self.wandb_group)
        self._training(track_run)

    def _training(self, track_run):
        '''Run constructed rl training with wandb'''
        if not os.path.exists(self.path_model_data):

            # create model
            model_handler = ModelHandlerAC(self.config,self.path_model_data,self.device)
            model_handler.create_model()

            # start algorithm
            sac = Sac(self.config,model_handler)

            # train and save model
            sac.train()

            track_run.finish()
            torch.cuda.empty_cache()
        else:
            print('Model loading not implemented')
