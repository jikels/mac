# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb

from mbmrl_torch.neural_networks.model_handler.handler_famle import FamleHandler
from mbmrl_torch.neural_networks.model_handler.handler_mb_vanilla import MbVanillaHandler
from mbmrl_torch.neural_networks.model_handler.handler_maml import MamlHandler
from mbmrl_torch.neural_networks.model_handler.handler_fomaml import FomamlHandler

# import utilities to collect data, process_data,init_models,
# train_models, record_videos, init cuda,..
from mbmrl_torch.utils.experiment_utilities import init_torch_cuda_set_device

# import Model predictive control untilities
from mbmrl_torch.reinforcement_learning.runners.constructor import mpc_online_adaption, mac_online_adaption

from mbmrl_torch.utils.rms_welford import NNNormalizer


class MetaTesting:
    '''
    Class for meta testing
    '''
    def run_meta_testing(self):
        '''
        Run constructed meta testing with wandb
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

        self.meta_testing(track_run)

    # 7. online adaption (meta-model testing)
    def meta_testing(self, track_run):
        '''
        Run constructed meta testing with wandb
        
        Args:
            track_run: wandb run
        
        '''

        # Initialize before online adaption
        self.device = init_torch_cuda_set_device(
            seed=self.config["seed"], cuda=self.config["cuda"], device_number=0)

        # init model handler and load model
        if self.algorithm == "famle":
            model_handler = FamleHandler(
                self.config,
                self.path_model_data,
                self.path_task_data,
                self.device,
                self.configs_training_tasks)
        elif self.algorithm == "fomaml":
            model_handler = FomamlHandler(
                self.config,
                self.path_model_data,
                self.path_task_data,
                self.device)
        elif self.algorithm == "maml":
            model_handler = MamlHandler(
                self.config,
                self.path_model_data,
                self.path_task_data,
                self.device)
        elif self.algorithm == "mb_vanilla":
            model_handler = MbVanillaHandler(
                self.config,
                self.path_model_data,
                self.path_task_data,
                self.device)
        else:
            print("define valid algorithm name (famle, fomaml, maml)")

        model_handler.load_model()

        # Init Controller
        if self.config["controller_type"] == "mpc":
            online_adaption = mpc_online_adaption
        elif self.config["controller_type"] == "mac":
            online_adaption = mac_online_adaption
        elif self.config["controller_type"] == "sac":
            print("not implemented: online_adaption = sac_online_adaption")
            quit()
        else:
            print("choose available controller type")
            quit()

        # run online adaption
        test = online_adaption(
            self.config,
            self.configs_training_tasks,
            self.configs_testing_tasks,
            model_handler)
        test.run()
        # test.run_test_sac(controller)
        track_run.finish()
        torch.cuda.empty_cache()
