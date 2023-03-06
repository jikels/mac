# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb

from mbmrl_torch.neural_networks.model_handler.handler_mb_vanilla import MbVanillaHandler

# import utilities to collect data, process_data,init_moels,
# train_models, record_videos, init cuda,..
from mbmrl_torch.utils import experiment_utilities

# import Model predictive control untilities
#from mbmrl_torch.reinforcement_learning.policies.MPC import mpc_online_adaption
from mbmrl_torch.reinforcement_learning.runners.constructor import mpc_online_adaption, mac_online_adaption

class MBTesting:
    '''
    Class for mbrl testing
    '''
    def run_mb_testing(self):
        '''
        Run constructed mbrl testing with wandb
        '''
        # init wandb run meta_testing
        track_run = wandb.init(
            project=self.config["env_name"],
            entity=self.config["wandb_entity"],
            reinit=True,
            config=self.wandb_config,
            job_type="Meta Testing",
            name=self.config["exp_name"],
            group="train_test_" + self.algorithm)

        # self.testing_sweep_params() #set parameters of current run during a
        # sweep

        self.mb_testing(track_run)

    # 7. online adaption (meta-model testing)
    def mb_testing(self, track_run):
        '''
        Run constructed meta testing with wandb
        
        Args:
            track_run: wandb run
        
        '''

        # Initialize before online adaption
        self.device = experiment_utilities.init_torch_cuda_set_device(
            seed=self.config["seed"], cuda=True, device_number=0)

        # init model handler and load model
        if self.algorithm == "mb_vanilla":
            model_handler = MbVanillaHandler(
                self.config,
                self.path_model_data,
                self.path_task_data,
                self.device)
        else:
            print("define valid algorithm name")

        model_handler.load_model()

        # Init Controller
        if self.config["controller_type"] == "mpc":
            online_adaption = mpc_online_adaption
        elif self.config["controller_type"] == "sac":
            print("not implemented: online_adaption = sac_online_adaption")
            quit()
        else:
            print("choose available controller type")
            quit()

        # run online adaption
        test = online_adaption(
            config=self.config,
            configs_testing_tasks=None,
            configs_training_tasks=None,
            model_handler=model_handler,)
        test.run()
        # test.run_test_sac(controller)
        track_run.finish()
        torch.cuda.empty_cache()