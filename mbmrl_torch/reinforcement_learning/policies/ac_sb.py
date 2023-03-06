from .base import Policy
import torch
from stable_baselines3.sac.sac import SAC
#from mbmrl_torch.benchmark.code.sac.sac import SAC
from mbmrl_torch.gym.utils.env_init import init_env

class ActorCriticPolicy(Policy):
    def __init__(self, ac_model_path):
        super().__init__()
        self.model = SAC.load(ac_model_path, env=None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_action(self,obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def _get_actions(self,obs):
        actions, states = self.model.predict(obs.cpu(), deterministic=True)
        return torch.from_numpy(actions).to(self.device)
    
    def get_action(self, observation):
        with torch.no_grad():
            a = self._get_action(observation)
            return a, {}
    
    def get_actions(self, observation):
        # return batch of actions (e.g., in MAC)
        return self._get_actions(observation)