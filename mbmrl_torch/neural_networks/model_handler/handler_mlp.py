import torch
from ..mlp import MLP as MLP_model
from mbmrl_torch.neural_networks.model_handler.base import HandlerBase
from ..model_trainer.mlp_trainer import MLPTrainer
import copy

class MLP_handler(HandlerBase):
    
    def __init__(self, config, path_model_data, path_task_data, device):
        super().__init__(config, path_model_data, path_task_data, device)

    def _create_model(self):
        model = MLP_model(
            dim_in=self.config["dim_in"],
            hidden=self.config["hidden_layers"],
            dim_out=self.config["state_dim"],
            cuda=self.config["cuda"],
            seed=self.config["seed"],
            dropout=self.config["dropout"],
            activation=self.config["hidden_activation"])
        self.model = model

    def _load_model(self):
        model_data = torch.load(self.path_model_data, map_location=self.device)
        self.model = MLP_model(**model_data["kwargs"])
        self.model.load_state_dict(model_data["state_dict"])
        self.model.data_mean_input = model_data["stats"]["data_mean_input"]
        self.model.data_std_input= model_data["stats"]["data_std_input"]
        self.model.data_mean_output= model_data["stats"]["data_mean_output"]
        self.model.data_std_output= model_data["stats"]["data_std_output"]
        self.model.to(self.device)
        print("\nLoaded dynamics model on ", self.device)

        if self.config["ensemble"] > 1:
            print("todo: implement ensemble training")
            quit()
        else:
            self.model = copy.deepcopy(self.model)
            self.online_model = copy.deepcopy(self.model)
    
    def _init_online_training(self):
        self.normalizer = self._init_normalizer(None,None) # init normalizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = None
        self.online_trainer = MLPTrainer(self.online_model, self.optimizer, scheduler)
        print("Online training initialized")
    
    def _train_model_online(self, train_in, train_out):

        # copy online model to start from scratch (k-step adaption)
        self.online_model = copy.deepcopy(self.model)
        # reset normalizer
        self.normalizer.reset_to_init()
        # calculate new statistics
        for i in range(len(train_in)):
            self.normalizer.update_stats(train_in[i], train_out[i])
        # get new statistics
        normalization_stats = self.normalizer.get_stats()
        # set statistics
        self.online_model.set_stats(normalization_stats)
        
        if self.config["ensemble"]==1:
            mean_loss = self.online_trainer.train(
                epochs=self.config["epoch"],
                data_in=train_in,
                data_out=train_out,
                batch_size=self.config["minibatch_size"],
                train_online=True) 
        else:
            print("todo: implement ensemble training")
            quit()
        return mean_loss