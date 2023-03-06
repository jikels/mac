# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#from mbmrl_torch.benchmark.load_sac import pretrained_SAC
from mbmrl_torch.utils.model_hook import hook
from mbmrl_torch.reinforcement_learning.policies import base
import torch
import numpy as np
import time
from einops import repeat
import wandb
from sklearn.decomposition import PCA

class MAC(base.Policy):
    
    def __init__(self,config, reference_policy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_policy = reference_policy
        self.lb = config["lb"]
        self.ub = config["ub"]
        self.embedding_dim = config["embedding_size"]
        self.popsize = config["popsize"]
        self.alpha = config["alpha"]
        self.horizon = config["horizon"]
        self.action_dim = config["action_dim"]
        self.discount = config["discount"]
        self.elites = config["elites"]
        self.mac_elites = config["mac_elites"]
        self.init_mean = torch.zeros(self.action_dim).to(self.device)
        self.init_std = 1
        self.state_dim = config["state_dim"]
        self.log_additional_data = config["log_additional_data"]
        self.reference_task = config["reference_task"]

        if self.log_additional_data == True:
            self.plots = log_mac_plots()

    def get_action(
        self,
        env,
        state,
        online_model,
        meta_model,
        step,
        reward,
        task_index,
        last_frame):
        return self.plan_action(
                        env,
                        state,
                        online_model,
                        meta_model,
                        step,
                        reward,
                        task_index,
                        last_frame)

    def plan_action(
        self,
        env,
        start_state,
        online_model,
        meta_model,
        step,
        prev_reward,
        prev_most_likely_task,
        prev_frame):
        start_time=time.time()

        if self.log_additional_data and step > 1:
            self.plots.log(
                step-1,
                self.embedding_output,
                prev_reward,
                self.prev_most_likely_task,
                self.prev_frame,
                self.action,
                self.action_ref)
            
            #if step:
                #self.log_mac_plots.make_mac_plots()

        self.online_model = online_model
        self.meta_model = meta_model
        self.env = env
        self.init_state = torch.from_numpy(start_state).float()
        #self.reference_policy.update_stats(start_state)
        self.prev_frame = prev_frame
        self.prev_most_likely_task = prev_most_likely_task
        
        self.obtain_solution()

        self.time_per_controller_step = time.time() - start_time

        self.log(
            step,
            self.action,
            self.action_samples.cpu().detach().numpy(),
            self.expected_reward,
            self.time_per_controller_step,
            self.best_similarity,
            self.action_ref)

        return self.action, self.next_states_0
        
    def predict_action_outcome_train_task(self, start_state, action):
        model_input = torch.cat((start_state, action), dim=1)
        next_state = self.meta_model[self.reference_task].predict(model_input)+start_state #de-normalized model output (=env state)
        return next_state

    def predict_action_outcome_test_task_online_model(self, start_state, action):
        hook_embedding = [hook(list(self.online_model._modules.items())[2][1])]
        model_input = torch.cat((start_state, action), dim=1)
        next_state = self.online_model.predict(model_input)+start_state #de-normalized model output (=env state)
        return next_state, hook_embedding[0].output

    def get_actions_pert(self, actions_ref):
        mean= torch.mean(actions_ref, dim=0)
        self.init_mean = self.init_mean * self.alpha + (1 - self.alpha) * mean
        a = torch.full_like(self.init_mean,-1)
        c = torch.ones_like(self.init_mean)
        lamb = torch.full_like(self.init_mean,4)
        r = c - a
        alpha = 1 + lamb * (self.init_mean - a) / r
        beta = 1 + lamb * (c - self.init_mean) / r
        m = torch.distributions.beta.Beta(alpha, beta)
        actions = a + m.sample_n(1024) * r
        return actions

    def get_actions_normal(self,action_elites):
        try:
            mean = torch.mean(action_elites, dim=0)
            std = torch.std(action_elites, dim=0)
            self.init_mean = self.init_mean * self.alpha + (1 - self.alpha) * mean
            self.init_std = self.init_std * self.alpha + (1 - self.alpha) * std
        except:
            pass
        
        m = torch.distributions.normal.Normal(self.init_mean,self.init_std)
        actions = m.sample_n(1024)

        return actions

    def get_actions_uniform(self):
        m = torch.distributions.uniform.Uniform(torch.squeeze(torch.full((1,8),-1.0)),torch.squeeze(torch.full((1,8),1.0)))
        actions = m.sample_n(1024).to(self.device)
        return actions

    def obtain_solution(self):
        costs_per_step, similarities, embeddings, samples, actions_ref, next_states_0 = self.cost_function()
        
        costs = np.sum(costs_per_step,axis=0)
        #indice = np.argmin(costs)
        similarity = np.sum(similarities,axis=0)
        #indice = np.argmin(similarity)

        stacked = np.stack((similarity,costs),axis=-1)
        
        indices = np.lexsort((stacked[:,1],stacked[:,0]))

        indice = indices[0]

        elites = indices[0:self.mac_elites]

        mean= torch.mean(torch.from_numpy(samples[elites,self.action_dim: self.action_dim + self.action_dim]), dim=0).to(self.device)
        self.init_mean = self.init_mean * self.alpha + (1 - self.alpha) * mean

        self.action = samples[indice][0:self.action_dim]
        self.action_ref = actions_ref[indice][0:self.action_dim]
        self.expected_reward = -1*costs_per_step[0,indice]
        self.best_similarity = similarity[indice]
        self.embedding_output = np.expand_dims(embeddings[indice][0:self.embedding_dim],axis=0)
        self.next_states_0 = next_states_0[indices[0]]
        

    def cost_function(self):
        self.action_samples = torch.FloatTensor(torch.zeros((self.popsize,self.horizon*self.action_dim))).to(self.device)#[popsize[action_dim*horizon]]
        self.actions_ref = torch.FloatTensor(torch.zeros((self.popsize,self.horizon*self.action_dim))).to(self.device)#[popsize[action_dim*horizon]]
        self.next_states_0 = torch.FloatTensor(torch.zeros((self.popsize,self.state_dim))).to(self.device) # [popsize[action_dim*horizon]]
        init_states =  torch.FloatTensor(torch.tile(input=self.init_state, dims = (self.popsize,1))).to(self.device) #[popsize[statesize]]
        
        embeddings = torch.FloatTensor(torch.zeros((self.popsize,self.embedding_dim*self.horizon))).to(self.device) #[popsize[horizon*embeddingsize]]

        all_costs = torch.FloatTensor(torch.zeros((self.horizon,self.popsize))).to(self.device)#[popsize[horizon]]
        similarities = torch.FloatTensor(torch.zeros((self.horizon,self.popsize))).to(self.device)
        

        #define number of batches and samples per batch
        n_batch = max(1, int(len(self.action_samples)/1024))
        per_batch = len(self.action_samples)/n_batch

        #predict rewards for batch of action with horizon h
        for i in range(n_batch):
            start_index = int(i*per_batch)
            end_index = len(self.action_samples) if i == n_batch - \
                1 else int(i*per_batch + per_batch)

            action_batch = self.action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]

            h=0
            for h in range(self.horizon):
                #get reference actions
                actions_train_task = self.reference_policy.get_actions(start_states) #[popsize[actionsize]]
                next_states_train_task = self.predict_action_outcome_train_task(start_states,actions_train_task) #[popsize[statesize]]
                self.actions_ref[start_index:end_index, h*self.action_dim: h * self.action_dim + self.action_dim]+= torch.squeeze(actions_train_task)

                #generate actions based on reference actions]
                #actions = self.get_actions_uniform()
                #actions = self.get_actions_normal(actions_train_task)
                actions = self.get_actions_pert(actions_train_task) 
            
                #predict next state in test task
                next_states_test_task, embedding_outputs = self.predict_action_outcome_test_task_online_model(start_states,actions)
                
                embeddings[start_index:end_index, h*self.embedding_dim: h * self.embedding_dim + self.embedding_dim] += torch.squeeze(embedding_outputs)

                #calculate cosine similarity
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                similarity =cos(next_states_test_task, next_states_train_task)

                #get indices that sort by similarity
                _, indices = torch.sort(similarity,descending=True)
                indices = indices[0:self.mac_elites]
                
                #predict reward based on cost function from environment
                predicted_reward = self.env.get_reward_tensor(actions[indices], start_states[indices],next_states_test_task[indices])

                #repeat best actions and update states
                actions = repeat(actions[indices], 'h w -> (repeat h) w', repeat=int(per_batch/self.mac_elites))
                self.action_samples[start_index:end_index][:, h*self.action_dim: h * self.action_dim + self.action_dim] = 0
                self.action_samples[start_index:end_index][:, h*self.action_dim: h * self.action_dim + self.action_dim] += actions

                #repeat best next states
                next_states_test_task = repeat(next_states_test_task[indices], 'h w -> (repeat h) w', repeat=int(per_batch/self.mac_elites))

                #repeat reward
                predicted_reward = repeat(predicted_reward, 'h -> (repeat h)', repeat=int(per_batch/self.mac_elites))

                #repeat similarity
                similarity = repeat(similarity[indices], 'h -> (repeat h)', repeat=int(per_batch/self.mac_elites))
                
                #add negative rewards (costs) and similarities to the cost function for evaluation of the best action sequence
                all_costs[h,start_index: end_index] += torch.neg(predicted_reward*self.discount**h)
                similarities[h,start_index: end_index] += torch.neg(similarity)

                start_states = next_states_test_task

                if h == 0:
                    self.next_states_0[start_index:end_index] += next_states_test_task

                h=h+1
    
        return all_costs.cpu().detach().numpy(),similarities.cpu().detach().numpy(),embeddings.cpu().detach().numpy(),self.action_samples.cpu().detach().numpy(),self.actions_ref.cpu().detach().numpy(),self.next_states_0.cpu().detach().numpy()

    def log(self, step, action, actions, expected_reward, time_per_controller_step,best_similarity, action_ref):
        log = {
            'expected_reward_per_step': expected_reward, 
            'time_per_controller_step':time_per_controller_step, 
            'action_samples':wandb.Histogram(actions),
            'action_samples_std': np.std(actions),
            'action_samples_mean': np.mean(actions),
            'action_executed_std': np.std(action),
            'action_executed_mean': np.mean(action),
            'selected_action':wandb.Histogram(action),
            'best_similarity': best_similarity,
            'action_reference_std': np.std(action_ref),
            'action_reference_mean': np.mean(action_ref)}
        wandb.log(log, step=step)
    
class log_mac_plots(object):

    def __init__(self):
        '''
        class that creates a pca table to analyze embeddings
        '''
        self.embedding_data = []
        self.reward_data = []
        self.most_likely_task = []
        self.step = []
        self.frames = []
        self.action_ex = []
        self.action_ref = []

    def log(self, step, embedding_data, reward_data, most_likely_task, frame, action_ex, action_ref):
        self.embedding_data.append(embedding_data)
        self.reward_data.append(reward_data)
        self.most_likely_task.append(most_likely_task)
        self.step.append(step)
        self.frames.append(wandb.Image(frame))
        self.action_ex.append(action_ex)
        self.action_ref.append(action_ref)

    def make_mac_plots(self):
        #pca embeddings
        features = np.concatenate(self.embedding_data, axis = 0)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        x_values = pca_result[:,0].tolist()
        y_values = pca_result[:,1].tolist()

        data = [[x,y,r,t,s,f] for (x,y,r,t,s,f) in zip(x_values, y_values,self.reward_data,self.most_likely_task,self.step,self.frames)]
        table = wandb.Table(data=data, columns = ["embedding_pc1", "embedding_pc2","reward","most_likely,task","step", "picture"])
        wandb.log({"outputs_embedding_pca": table})

        #pca actions
        features = np.concatenate((self.action_ex,self.action_ref), axis = 0)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        x_values = pca_result[:,0].tolist()
        y_values = pca_result[:,1].tolist()

        #create type array
        length = np.size(self.action_ref, 0)
        ex = np.repeat("executed", length)
        ref = np.repeat("reference", length)
        type = np.concatenate((ex,ref), axis = 0)

        step = np.concatenate((self.step,self.step), axis = 0)
        reward = np.concatenate((self.reward_data,self.reward_data), axis = 0)

        data = [[s,x,y,t,r] for (s,x,y,t,r) in zip(step,x_values,y_values,type,reward)]
        table = wandb.Table(data=data, columns = ["step", "action_pc1", "action_pc2", "action_type","reward"])
        wandb.log({"actions_ex_ref_pca": table})

        data = [[ae,ar,r,s] for (ae,ar,r,s) in zip(self.action_ex,self.action_ref,self.reward_data, self.step)]
        table = wandb.Table(data=data, columns = ["action_executed", "action_reference","reward","step"])
        wandb.log({"actions": table})
