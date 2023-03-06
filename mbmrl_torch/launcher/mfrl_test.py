# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
import os
import numpy as np
from mbmrl_torch.neural_networks.model_handler.handler_actor_critic import ModelHandlerAC
from mbmrl_torch.reinforcement_learning.runners.constructor import SACTest

# import utilities to collect data, process_data,init_moels,
# train_models, record_videos, init cuda,..
from mbmrl_torch.utils import data_processing
from mbmrl_torch.utils import experiment_utilities

class MFRLTesting:
    '''Class for model free reinforcement learning testing'''
    def run_testing(self):
        '''Run constructed rl testing with wandb'''
        # init wandb run meta_testing
        track_run = wandb.init(
            project=self.config["env_name"],
            entity=self.config["wandb_entity"],
            reinit=True,
            config=self.wandb_config,
            job_type="Meta Testing",
            name=self.config["exp_name"],
            group="train_test_" + self.algorithm)

        self.testing(track_run)

    # 7. online adaption (meta-model testing)
    def testing(self, track_run):
        '''Run constructed rl testing with wandb'''
        # Initialize before online adaption
        self.device = experiment_utilities.init_torch_cuda_set_device(
            seed=self.config["seed"], cuda=True, device_number=0)

        # init model handler and load model
        if self.algorithm == "sac":
            model_handler = ModelHandlerAC(
                self.config, self.path_model_data, self.device)
        else:
            print("define valid algorithm name (sac)")
        
        model_handler.load_model()

        # if task settings are given to test
        # performance on different tasks
        path_task_data = self.config["task_data"]
        if os.path.exists(path_task_data):
            print('loading existing task configs')
            configs_testing_tasks= np.load(
                path_task_data+"/config_testing_tasks.npy",
                allow_pickle=True)
            configs_training_tasks=np.load(
                path_task_data+"/config_training_tasks.npy",
            allow_pickle=True)
        else:
            configs_training_tasks = None
            configs_testing_tasks = None


        # run online adaption
        test = SACTest(
            self.config,
            model_handler,
            configs_training_tasks,
            configs_testing_tasks)
        test.run()
        # test.run_test_sac(controller)
        track_run.finish()
        torch.cuda.empty_cache()
