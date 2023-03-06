# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .model_adaption import OnlineModelAdaption
from .env_runner import Runner

from ..policies.mpc import MPC
from ..policies.mac import MAC
from ..policies.ac_sb import ActorCriticPolicy as ac

from mbmrl_torch.neural_networks.model_handler.handler_actor_critic import ModelHandlerAC

class mpc_online_adaption(OnlineModelAdaption):
    
    """Superclass for MPC online adaption"""

    def __init__(
        self,
        config,
        configs_training_tasks,
        configs_testing_tasks,
        model_handler
    ):
        
        self.policy = MPC(config)

        super(mpc_online_adaption, self).__init__(
            config=config,
            configs_training_tasks=configs_training_tasks,
            configs_testing_tasks=configs_testing_tasks,
            model_handler=model_handler
        )

    def get_action(self, state):
        return self.policy.plan_action(
                    env=self.env,
                    model=self.model_handler.return_online_model(),
                    init_state=state,
                    step=self.step)

class mac_online_adaption(OnlineModelAdaption):
    
    """Superclass for MAC online adaption"""

    def __init__(
        self,
        config,
        configs_training_tasks,
        configs_testing_tasks,
        model_handler
    ):  
        # initialize actor critic
        # model handler
        #ac_handler = ModelHandlerAC(config, config["mac_reference_policy"])
        # load existing actor critic model
        #ac_handler.load_model()
        #ac_model = ac_handler.model
        #ac_normalizer = ac_handler.normalizer
        # wrap actor critic model as policy 
        #ac_policy = ac(ac_model,ac_normalizer)
        ac_policy = ac(config["mac_reference_policy"])

        # initialize mac
        config['embedding_size'] = model_handler.model.embedding_dim
        self.policy = MAC(config, reference_policy = ac_policy)

        super(mac_online_adaption, self).__init__(
            config=config,
            configs_training_tasks=configs_training_tasks,
            configs_testing_tasks=configs_testing_tasks,
            model_handler=model_handler
        )

    def get_action(self, state):
        return self.policy.plan_action(
                        self.env,
                        state,
                        self.model_handler.return_online_model(),
                        self.model_handler.return_meta_model(),
                        self.step,
                        self.reward,
                        self.model_handler.task_index,
                        self.recorder.get_last_frame())

class SACTest(Runner):
    
    """Superclass for sac testing"""

    def __init__(
        self,
        config,
        model_handler,
        configs_training_tasks,
        configs_testing_tasks
    ):
        
        model = model_handler.model
        normalizer = model_handler.normalizer
        self.policy = ac(model, normalizer)

        super(SACTest, self).__init__(
            config=config,
            configs_training_tasks=configs_training_tasks,
            configs_testing_tasks=configs_testing_tasks,
        )

    def get_action(self, state):
        return self.policy.get_action(state)