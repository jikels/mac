# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import time
from ...utils.rotation import quaternion_to_euler
import wandb

from .base import Policy

class MPC(Policy):
    def __init__(self, config):
        #update regularly
        self.model = None
        self.env = None
        self.init_state = None

        #only init once
        self.config = config
        self.action_dim = self.config["action_dim"]
        self.state_dim = self.config["state_dim"]
        self.horizon = self.config["horizon"]
        self.discount = self.config["discount"]
        self.lb = self.config["lb"]
        self.ub = self.config["ub"]
        self.popsize = self.config["popsize"]
        self.elites = int(round(self.config["elites"]*self.popsize))
        self.alpha = self.config["alpha"]
        self.epsilon = 1e-8
        #self.critic = pretrained_SAC()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_mean = torch.zeros(self.action_dim).to(self.device)
        self.init_std = torch.ones(self.action_dim).to(self.device)
        self.mpc_type = self.config["mpc_type"]

        if self.config["use_CEM"]:
            #self.action_sampler = self.get_actions_pert
            self.action_sampler = self.get_actions_normal
        else:
            self.action_sampler = self.get_actions_uniform

    def get_action(
        self,
        env,
        model,
        state,
        step):
        return self.plan_action(
                    env,
                    model,
                    state,
                    step)
    
    def plan_action(self,env, model,init_state, step):
        #update regularly
        self.model = model
        self.env = env
        self.init_state = init_state

        actions_argmin, expected_reward, expected_next_state, time_per_MPC_step = self.obtain_solution() #action sequences ordered by reward
        
        actions = actions_argmin.cpu().detach().numpy()
        action = actions[0][0:self.action_dim]
        expected_reward = expected_reward.cpu().detach().numpy()
        expected_next_state = expected_next_state.cpu().detach().numpy()
        
        self.log(step, action, actions, expected_reward, time_per_MPC_step)
        
        return action, expected_next_state
        
    def obtain_solution(self):

        start_time = time.time()
        self.cost_fn()
        costs = torch.sum(self.all_costs,axis=0)
        
        if  self.mpc_type == "normal":
            indices = torch.argsort(costs)
            #print(indices)
        elif self.mpc_type == "critic":
            indices = torch.argsort(costs)
        elif self.mpc_type == "anchor":
            trust = torch.sum(self.reward_trusted_out,axis=0)
            stacked = torch.stack((trust,costs),axis=-1).cpu().detach().numpy()
            indices = np.lexsort((stacked[:,1],stacked[:,0]))
        else:
            print("define valid mpc type (normal, anchor, critic)")
            quit()
        
        if self.config["use_CEM"]:
            elites = indices[0:self.elites]
            mean = torch.mean(self.action_samples[elites,0:self.action_dim], dim=0).to(self.device)
            std = torch.std(self.action_samples[elites,0:self.action_dim], dim=0).to(self.device)
            self.init_mean = self.init_mean * self.alpha + (1 - self.alpha) * mean
            self.init_std = self.init_std * self.alpha + (1 - self.alpha) * std
        
        return self.action_samples[indices], -1*self.all_costs[0,indices[0]], self.next_states_0[indices[0]], time.time() - start_time 

    def cost_fn(self):
        
        #inititialze Tensors
        self.action_samples = torch.FloatTensor(
            torch.zeros((self.popsize,self.horizon*self.action_dim))).to(self.device) # [popsize[action_dim*horizon]]
        self.next_states_0 = torch.FloatTensor(
            torch.zeros((self.popsize,self.state_dim))).to(self.device) # [popsize[action_dim*horizon]]
        init_states = torch.FloatTensor(
            np.repeat([self.init_state], len(self.action_samples), axis=0)).to(self.device) \
                if self.model.cuda_enabled else torch.FloatTensor(np.repeat([self.init_state], len(self.action_samples), axis=0)) #[popsize[state_dim]]
        self.reward_trusted_out = torch.FloatTensor(
            np.zeros((self.horizon,len(self.action_samples)))).to(self.device)

        self.all_costs = torch.FloatTensor(
            np.zeros((self.horizon,len(self.action_samples)))).to(self.device) \
                if self.model.cuda_enabled else torch.FloatTensor(
            np.zeros((self.horizon,len(self.action_samples))))

        #define number of batches and action_samples per batch
        n_batch = max(1, int(len(self.action_samples)/1024))
        per_batch = len(self.action_samples)/n_batch
        
        #predict rewards for batch of action with horizon h
        for i in range(n_batch):
            start_index = int(i*per_batch)
            end_index = len(self.action_samples) if i == n_batch - \
                1 else int(i*per_batch + per_batch)

            start_states = init_states[start_index:end_index]

            h=0
            for h in range(self.horizon):
                # sample an action batch, concatenate model
                # input and predict next states
                actions = self.action_sampler()
                model_input = torch.cat((start_states, actions), dim=1)
                next_states = self.model.predict(model_input)+start_states #de-normalized model output

                # predict reward based on cost function from environment
                predicted_reward = self.env.get_reward_tensor(actions, start_states,next_states)

                if self.config["mpc_type"] == "anchor":
                    self.anchor(h,start_index,end_index,predicted_reward,next_states,anchor=[5,-5,1.0,0.4,20,-20,20,-20,20,-20])
                elif self.config["mpc_type"] == "critic":
                    print("critic not implemented")
                    quit()
                    q_values = self.critic.get_q_values(start_states,actions)/100
                    # add negative rewards (costs) to the cost function for evaluation of the best action sequence
                    self.all_costs[h,start_index: end_index] += torch.neg((predicted_reward+q_values.squeeze())*self.discount**h)
                elif self.config["mpc_type"] == "normal":
                    # add negative rewards (costs) to the cost function for evaluation of the best action sequence
                    self.all_costs[h,start_index: end_index] += torch.neg(predicted_reward*self.discount**h)

                self.action_samples[start_index:end_index][:, h*self.action_dim: h * self.action_dim + self.action_dim] = 0
                self.action_samples[start_index:end_index][:, h*self.action_dim: h * self.action_dim + self.action_dim] += actions

                if h == 0:
                    self.next_states_0[start_index:end_index] += next_states

                start_states = next_states

                h=h+1

    def anchor(self,h,start_index,end_index,predicted_reward,next_states,anchor):
        #predict reward based on reward model or cost function from environment
        #calculate euler tensor
        euler = quaternion_to_euler(next_states[:,3:7])

        #find not trusted regions
        y_posafter_cost = torch.squeeze(
            torch.where(torch.logical_and(next_states[:,1]<=anchor[0], next_states[:,1]>=anchor[1]),0,1))
        z_posafter_cost = torch.squeeze(
            torch.where(torch.logical_and(next_states[:,2]<=anchor[2], next_states[:,2]>=anchor[3]),0,1)) #xyz xyzw
        x_euler_posafter_cost = torch.squeeze(
            torch.where(torch.logical_and(euler[:,0]<=anchor[4], euler[:,0]>=anchor[5]),0,1))
        y_euler_posafter_cost = torch.squeeze(
            torch.where(torch.logical_and(euler[:,1]<=anchor[6], euler[:,1]>=anchor[7]),0,1))
        z_euler_posafter_cost = torch.squeeze(
            torch.where(torch.logical_and(euler[:,2]<=anchor[8], euler[:,2]>=anchor[9]),0,1))

        not_trusted = y_posafter_cost+z_posafter_cost+x_euler_posafter_cost+y_euler_posafter_cost+z_euler_posafter_cost

        reward_trusted = torch.where(
            not_trusted>0,torch.ones_like(predicted_reward),torch.zeros_like(predicted_reward)) #remember_later = assign less reward instead of 0
    
        #add negative rewards (costs) to the cost function for evaluation of the best action sequence
        self.all_costs[h,start_index: end_index] += torch.neg(predicted_reward*self.discount**h)
        self.reward_trusted_out[h,start_index: end_index] += reward_trusted*self.discount**h

    def get_actions_pert(self):
        a = torch.full_like(self.init_mean,-1)
        c = torch.ones_like(self.init_mean)
        lamb = torch.full_like(self.init_mean,4)
        r = c - a
        alpha = 1 + lamb * (self.init_mean - a) / r
        beta = 1 + lamb * (c - self.init_mean) / r
        m = torch.distributions.beta.Beta(alpha, beta)
        actions = a + m.sample_n(1024) * r
        return actions

    def get_actions_normal(self):
        m = torch.distributions.normal.Normal(self.init_mean,self.init_std)
        actions = m.sample_n(1024)
        actions = torch.clamp(actions,-1,1)
        return actions

    def get_actions_uniform(self):
        m = torch.distributions.uniform.Uniform(
            torch.squeeze(torch.full((1,self.action_dim),self.lb,dtype=torch.float32)),
            torch.squeeze(torch.full((1,self.action_dim),self.ub,dtype=torch.float32))
            )
        actions = m.sample_n(1024).to(self.device)
        actions[:1,:self.action_dim] = 0
        return actions
    
    def log(self, step, action, actions, expected_reward, time_per_controller_step):
        log = {
            'expected_reward_per_step': expected_reward, 
            'time_per_controller_step':time_per_controller_step, 
            'action_samples':wandb.Histogram(actions),
            'action_samples_std': np.std(actions),
            'action_samples_mean': np.mean(actions),
            'action_executed_std': np.std(action),
            'action_executed_mean': np.mean(action),
            'selected_action':wandb.Histogram(action)}
        wandb.log(log, step=step)