# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from mbmrl_torch.launcher import constructor
import mbmrl_torch.launcher.alg_mapping as rl_type
import os
import mbmrl_torch

import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# get absolute path to project data dir 
config_data_dir = os.path.join(
        os.path.abspath(mbmrl_torch.__file__)[:-11],'configurations', 'conf')

@hydra.main(config_path=config_data_dir,
            config_name="config", version_base=None)
def main(config):
    '''
    Main function to run the experiment
    
    Args:
        config: config file
    '''
    
    # define specific config
    if config["general"]["learning_alg"] == "famle":
        cfg = config["famle"]
    elif config["general"]["learning_alg"] == "fomaml":
        cfg = config["fomaml"]
    elif config["general"]["learning_alg"] == "maml":
        cfg = config["maml"]
    elif config["general"]["learning_alg"] == "mb_vanilla":
        cfg = config["mb_vanilla"]
    elif config["general"]["learning_alg"] == "sac":
        cfg = config["sac"]
    else:
        print("please define valid learning algorithm (famle,maml,fomaml, mb_vanilla, sac)")

    # set seeds for reproducibility
    set_seed(cfg["seed"])

    # todo: make modular constructor
    # Adjust run to rl type
    if config["general"]["learning_alg"] in rl_type.meta_rl():
        # define and run experiment type
        if config["general"]["experiment"] == "train_test":
            # init experiment with specific config
            e = constructor.MetaTrainTest(cfg, config["general"]["learning_alg"])
            e.run_meta_training()
            e.run_meta_testing()
        elif config["general"]["experiment"] == "train":
            cfg["save_meta_model"] = True
            # init experiment with specific config
            e = constructor.MetaTrain(cfg, config["general"]["learning_alg"])
            e.run_meta_training()
        elif config["general"]["experiment"] == "test":
            # init experiment with specific config
            e = constructor.MetaTest(cfg, config["general"]["learning_alg"])
            e.run_meta_testing()
        elif config["general"]["experiment"] == "train_sweep":
            cfg["model_data"] = None
            cfg["save_meta_model"] = False
            # init experiment with specific config
            e = constructor.MetaTrain(cfg, config["general"]["learning_alg"])
            e.run_meta_training()
        elif config["general"]["experiment"] == "test_sweep":
            cfg["record_video"] = False
            # init experiment with specific config
            e = constructor.MetaTest(cfg, config["general"]["learning_alg"])
            e.run_meta_testing()
        else:
            print("Please define valid experiment type: train_test, train, test, train_sweep, test_sweep")
            quit()
    elif config["general"]["learning_alg"] in rl_type.mb_rl():
        if config["general"]["experiment"] == "train":
            # init experiment with specific config
            e = constructor.MBMRLTrain(cfg, config["general"]["learning_alg"])
            e.run_mb_training()
        elif config["general"]["experiment"] == "test":
            # init experiment with specific config
            e = constructor.MBMRLTest(cfg, config["general"]["learning_alg"])
            e.run_mb_testing()
    elif config["general"]["learning_alg"] in rl_type.mf_rl():
        if config["general"]["experiment"] == "train":
            # init experiment with specific config
            e = constructor.MFRLTrain(cfg, config["general"]["learning_alg"])
            e.train()
        elif config["general"]["experiment"] == "train_sweep":
            cfg["model_data"] = None
            cfg["save_freq"] = None
            # init experiment with specific config
            e = constructor.MFRLTrain(cfg, config["general"]["learning_alg"])
            e.train()
        elif config["general"]["experiment"] == "test":
            # init experiment with specific config
            e = constructor.MFRLTest(cfg, config["general"]["learning_alg"])
            e.run_testing()


if __name__ == '__main__':
    main()
