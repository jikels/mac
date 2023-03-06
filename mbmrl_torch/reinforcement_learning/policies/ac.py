from .base import Policy
import torch

class ActorCriticPolicy(Policy):
    def __init__(self, ac_model, normalizer):
        super().__init__()
        self.model = ac_model
        self.normalizer = normalizer
    
    def update_stats(self, o):
        self.normalizer.update(o)
        statistics = self.normalizer.get_statistics()
        statistics=[torch.from_numpy(statistics[0]).cuda().float(),torch.from_numpy(statistics[1]).cuda().float()]
        self.model.set_stats(statistics)

    def _get_action(self, o):
        self.update_stats(o)
        o = torch.as_tensor(o, dtype=torch.float32).cuda()
        action = self.model.act(o,deterministic=True)
        return action
    
    def _get_actions(self, o):
        actions = self.model.act_tensor(o,deterministic=True)
        return actions
    
    '''def get_action(self, observation):
        with torch.no_grad():
            a = self._get_action(observation)
            return a, {}'''

    def get_action(self, observation):
        #action, _states = self.model.predict(observation, deterministic=True)
        action = self._get_action(observation)
        return action, {}
    
    def get_actions(self, observation):
        # return batch of actions (e.g., in MAC)
        return self._get_actions(observation)