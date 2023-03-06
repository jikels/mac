import torch
import os
from mbmrl_torch.utils.rms_welford import NNNormalizer

class HandlerBase:
    
    def __init__(self, config, path_model_data, path_task_data, device):
        self.config = config
        self.path_model_data = path_model_data
        self.path_task_data = path_task_data
        self.device = device

    def _init_model_weights(self):
        if isinstance(self.model, torch.nn.Linear):
            torch.nn.init.xavier_uniform(self.model.weight)
            self.model.bias.data.fill_(0.01)

    def _init_normalizer(self, tasks_in, tasks_out):
        # calculate and save normalization stats
        path_norm_stats = os.path.join(self.path_task_data, 'normalization_statistics.npy')
        normalizer = NNNormalizer(self.config["state_dim"],self.config["action_dim"])
        if not os.path.exists(path_norm_stats):
            normalizer.calculate_stats(tasks_in, tasks_out)
            normalizer.save(self.path_task_data)
            print("saved normalization statistics")
        else:
            normalizer.load(path_norm_stats)
            print("loaded normalization statistics")
        return normalizer

    def _init_offline_training(self, tasks_in, tasks_out):
        self.normalizer = self._init_normalizer(tasks_in,tasks_out) # init normalizer
        normalization_statistics = self.normalizer.get_stats() # get current normalization statistics
        self.model.set_stats(normalization_statistics) # set normalization statistics