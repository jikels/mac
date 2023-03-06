# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import wandb
import numpy as np

from mbmrl_torch.neural_networks.model_handler.handler_famle import FamleHandler
from mbmrl_torch.neural_networks.model_handler.handler_mb_vanilla import MbVanillaHandler
from mbmrl_torch.neural_networks.model_handler.handler_maml import MamlHandler
from mbmrl_torch.neural_networks.model_handler.handler_fomaml import FomamlHandler

from mbmrl_torch.utils import data_processing, data_collection, experiment_utilities

class MetaTraining:
    '''Class for meta training'''
    def run_meta_training(self):
        '''Run constructed meta training with wandb'''
        # init wandb run meta_training
        track_run = wandb.init(
            project=self.config["env_name"],
            entity=self.config["wandb_entity"],
            reinit=True,
            config=self.wandb_config,
            job_type=self.wandb_job_type_tr,
            name=self.config["exp_name"],
            group=self.wandb_group)
        self.meta_training(track_run)

    def meta_training(self, track_run):
        '''Run constructed meta training with wandb'''
        # 6. Create, train and save a meta model or load existing
        if not os.path.exists(self.path_model_data):
            
            # process to train set
            path_proc_train_set = os.path.join(self.path_task_data, 'processed_train_set.npz')
            if not os.path.exists(path_proc_train_set):
                tasks_in, tasks_out, high, low = data_processing.process_to_meta_training_set(
                    self.path_task_data)
                np.savez_compressed(
                    path_proc_train_set,
                    a=tasks_in,
                    b=tasks_out)
                print("save compressed copy of train set")
            else:
                print("loaded compressed copy of train set")
                loaded = np.load(path_proc_train_set)
                tasks_in = loaded['a']
                tasks_out = loaded['b']
                
            # Initialize Model handler
            if self.algorithm == "famle":
                model_handler = FamleHandler(
                    config = self.config,
                    path_model_data = self.path_model_data,
                    path_task_data = self.path_task_data,
                    device = self.device,
                    configs_training_tasks = self.configs_training_tasks)
                print("init embedding neural network and FAMLE algorithm")
            elif self.algorithm == "fomaml":
                print("init multi layer neural network and FOMAML algorithm")
                model_handler = FomamlHandler(
                    self.config,
                    self.path_model_data,
                    self.path_task_data,
                    self.device)
            elif self.algorithm == "maml":
                print("init multi layer neural network and MAML algorithm")
                model_handler = MamlHandler(
                    self.config,
                    self.path_model_data,
                    self.path_task_data,
                    self.device)
            elif self.algorithm == "mb_vanilla":
                print("init multi layer neural network")
                model_handler = MbVanillaHandler(
                    self.config,
                    self.path_model_data,
                    self.path_task_data,
                    self.device)
            else:
                print("define valid algorithm name (famle, fomaml, maml)")

            # create model
            model_handler.create_model()

            # init model weigths
            model_handler.init_model_weights()

            # train and save model
            with track_run:
                print("start training")
                model_handler.training(
                    tasks_in,
                    tasks_out,
                    track_run,
                    save=self.config["save_meta_model"])

            track_run.finish()
            torch.cuda.empty_cache()
