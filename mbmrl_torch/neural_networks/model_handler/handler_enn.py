import torch
import copy
from ..enn import EmbeddingNN as ENN_model
from mbmrl_torch.neural_networks.model_handler.base import HandlerBase
from mbmrl_torch.neural_networks.model_trainer.enn_trainer import ENNTrainer

class EnnHandler(HandlerBase):
    
    def __init__(self, config, path_model_data, path_task_data, device):
        super().__init__(config, path_model_data, path_task_data, device)

    def _create_model(self):
        model = ENN_model(
            dim_in=self.config["dim_in"],
            hidden=self.config["hidden_layers"],
            dim_out=self.config["state_dim"],
            embedding_dim=self.config["embedding_size"],
            num_tasks=len(self.configs_training_tasks),
            cuda=self.config["cuda"],
            seed=self.config["seed"],
            dropout=self.config["dropout"],
            activation=self.config["hidden_activation"])
        self.model = model

    def _load_model(self):
        model_data = torch.load(self.path_model_data, map_location=self.device)
        model_data["kwargs"]["cuda"] = self.config["cuda"]
        self.model = ENN_model(**model_data["kwargs"])
        self.model.load_state_dict(model_data["state_dict"])
        self.model.data_mean_input = model_data["other"]["data_mean_input"]
        self.model.data_std_input= model_data["other"]["data_std_input"]
        self.model.data_mean_output= model_data["other"]["data_mean_output"]
        self.model.data_std_output= model_data["other"]["data_std_output"]
        self.model.fix_task(model_data["other"]["fixed_task_id"])
        self.model.to(self.device)
        print("\n Loaded dynamics model on ", self.device)

        #5.1 duplicate models to estimate most likely embeddings
        with open(self.config["exp_resdir"] + "/costs.txt", "w+") as f:
            f.write("")
        
        if self.config["num_embeddings"]==0:
                #dynamic models
                self.models = [copy.deepcopy(self.model) for _ in range(len(self.configs_training_tasks))] #list of meta models for ease of use in mac and most likely embedding calculations
                #self.online_models = [copy.deepcopy(self.meta_model) for _ in range(len(configs_training_tasks))]

                self.online_model = copy.deepcopy(self.model)
                self.online_model.fix_task(0)
        else:
                #todo:extend embedding
                #dynamic models
                print("extend embedding not yet implemented")
                quit()
                #self.meta_models = [copy.deepcopy(self.meta_model) for _ in range(self.config["num_embeddings"])]
                #self.online_models = [copy.deepcopy(self.meta_model) for _ in range(self.config["num_embeddings"])]

        #5.3 assign task_id to each dynamics model
        
        ##list of meta models for ease of use in mac and most likely embedding calculations
        for task_id, m in enumerate(self.models):
            m.fix_task(task_id)

        #online models for online model adaption
        #for task_id, m in enumerate(self.online_models):
            #m.fix_task(task_id)
        
        #self.online_model = copy.deepcopy(self.online_models[0])

    def _init_online_training(self):
        self.normalizer = self._init_normalizer(None,None) # init normalizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = None
        self.online_trainer = ENNTrainer(self.online_model, self.optimizer, scheduler)
        print("Online training initialized")
       