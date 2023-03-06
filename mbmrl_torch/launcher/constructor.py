# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mbmrl_torch.launcher import meta_test
from mbmrl_torch.launcher import meta_train
from mbmrl_torch.launcher import mfrl_train
from mbmrl_torch.launcher import mfrl_test
from mbmrl_torch.launcher import mb_train
from mbmrl_torch.launcher import mb_test
from mbmrl_torch.launcher import base

class MetaTrain(base.Base, meta_train.MetaTraining):
    '''
    Constructor for Meta Training

    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)

class MetaTest(base.Base, meta_test.MetaTesting):
    '''
    Constructor for Meta Testing
    
    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)

class MetaTrainTest(base.Base, meta_train.MetaTraining, meta_test.MetaTesting):
    '''
    Constructor for Meta Training and Testing
    
    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)

class MFRLTrain(base.Base, mfrl_train.MFRLTraining):
    '''
    Constructor for Model-Free RL Training
    
    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)

class MFRLTest(base.Base, mfrl_test.MFRLTesting):
    '''
    Constructor for Model-Free RL Testing
    
    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)

class MBMRLTrain(base.Base, mb_train.MBTraining):
    '''
    Constructor Model-based RL Training
    
    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)

class MBMRLTest(base.Base, mb_test.MBTesting):
    '''
    Constructor for Model-based RL Testing
    
    Args:
        config: config file
        algorithm: algorithm to use
    '''
    def __init__(self, cfg, alg):
        super().__init__(config=cfg, algorithm=alg)
