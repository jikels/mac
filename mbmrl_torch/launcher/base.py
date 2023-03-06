# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mbmrl_torch.utils import experiment_utilities, data_collection
from mbmrl_torch.gym.utils.env_init import init_env
from mbmrl_torch.gym.utils.helper_functions import get_observation_dim, get_action_dim
from . import alg_mapping as rl_type
import omegaconf
import os

class Base:
    '''
    Base class as parent for all RL experiments
    '''
    def __init__(self, config, algorithm):
        '''
        Initialize the experiment
        
        Args:
            config: config file
            algorithm: algorithm to use
        '''
        print('init base...')

        # Initialize torch and cuda
        self.device = experiment_utilities.init_torch_cuda_set_device(
            seed=config["seed"],
            cuda=True,
            device_number=0)

        # Set algorithm name
        self.algorithm = algorithm

        # Create directories to track the experiment
        experiment_name, res_dir = experiment_utilities.init_experiment(config)
        config["exp_name"] = experiment_name
        config["exp_resdir"] = res_dir

        # Init env
        self.env = init_env(env_name=config["env_name"])

        # Get state and action dim and write to config (only works for identical envs!)
        config["state_dim"] = get_observation_dim(self.env)
        config["action_dim"] = get_action_dim(self.env)

        # Adjust config to rl type
        if self.algorithm in rl_type.meta_rl():
            config = self._config_meta_rl(config)
        elif self.algorithm in rl_type.mb_rl():
            config = self._config_mb_rl(config)
        elif self.algorithm in rl_type.mf_rl():
            config = self._config_mf_rl(config)

        # Create new model path or use existing
        if not os.path.exists(
                config["model_data_dir"] + "/" + config["env_name"]):
            os.makedirs(config["model_data_dir"] + "/" + config["env_name"])

        if config["model_data"] is None:
            # Set new model name
            config["model_name"] = config["model_name"] + "_" + experiment_name

            # Create path to save the model
            self.path_model_data = config["model_data_dir"] + "/" + \
                config["env_name"] + "/" + config["model_name"] + ".pt"
        else:
            self.path_model_data = config["model_data"]

        # set config in class and wandb
        self.config = config
        self.wandb_config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)

    def _config_meta_rl(self, config):
        '''
        Adjust config to meta RL

        Args:
            config: config file

        Returns:
            config: adjusted config file
        '''
        print('init meta rl...')
        # Define wandb descriptions
        self.wandb_job_type_tr = "Meta Training"
        self.wandb_job_type_ts = "Meta Testing"
        self.wandb_group = "train_test_" + self.algorithm

        if config["task_data"] is None:
            self.path_task_data = config["task_data_dir"] + \
                config["env_name"] + "/" + config["exp_name"]
        else:
            self.path_task_data = config["task_data"]

        # Create Meta Train, Test Set or load exisiting
        self.configs_testing_tasks, self.configs_training_tasks = experiment_utilities.data_gen_load(
            self.env,
            self.path_task_data,
            config)

        # Define dim_in for model
        config["dim_in"] = config['state_dim'] + config['action_dim']

        return config

    def _config_mb_rl(self, config):
        '''
        Adjust config to model-based RL

        Args:
            config: config file

        Returns:
            config: adjusted config file
        '''
        print('init mb rl...')
        # Define wandb descriptions
        self.wandb_job_type_tr = "Training"
        self.wandb_job_type_ts = "Testing"
        self.wandb_group = "train_test_" + self.algorithm

        if config["task_data"] is None:
            self.path_task_data = config["task_data_dir"] + \
                config["env_name"] + "/" + config["exp_name"]
        else:
            self.path_task_data = config["task_data"]

        # Create Meta Train, Test Set or load exisiting
        self.configs_testing_tasks, self.configs_training_tasks = experiment_utilities.data_gen_load(
            self.env,
            self.path_task_data,
            config)

        # Define dim_in for model
        config["dim_in"] = config['state_dim'] + config['action_dim']

        return config

    def _config_mf_rl(self, config):
        '''
        Adjust config to model-free RL
        
        Args:
            config: config file
        
        Returns:
            config: adjusted config file
        '''
        print('init mf rl...')
        self.wandb_job_type_tr = "Training"
        self.wandb_job_type_ts = "Testing"
        self.wandb_group = "train_test_" + self.algorithm
        return config