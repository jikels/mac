import torch

from mbmrl_torch.neural_networks.actor_critic import MLPActorCritic
from mbmrl_torch.utils.rms_welford import RunningMeanStd
import os

class ModelHandlerAC():
    
    def __init__(self, config, path_model_data, device=None):
        self.config = config
        self.path_model_data = path_model_data
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self):
        if os.path.exists(self.path_model_data)==False:
            self.model = MLPActorCritic(
                self.config["state_dim"],
                self.config["action_dim"],
                torch.ones(self.config["action_dim"]).to(self.device),
                self.config["hidden_sizes"]).to(self.device)
        else: print("Model already exists")
        
    def save_model(self):
        self.model.save(self.path_model_data)

    def load_model(self):
        model_data = torch.load(self.path_model_data, map_location=self.device)
        self.model = MLPActorCritic(**model_data["kwargs"])
        self.model.load_state_dict(model_data["state_dict"])
        self.model.pi_env_stats = model_data["others"]["env_stats_pi"]
        self.model.q1_env_stats = model_data["others"]["env_stats_q1"]
        self.model.q2_env_stats = model_data["others"]["env_stats_q2"]
        self.model.to(self.device)
        print("\nLoaded ac model on ", self.device)

        statistics=[self.model.pi_env_stats[0].detach().cpu().numpy(),self.model.pi_env_stats[1].detach().cpu().numpy(),self.model.pi_env_stats[2]]
        self.normalizer = RunningMeanStd((self.config['state_dim'],))
        self.normalizer.set_statistics([statistics[0], statistics[1], statistics[2]])
    