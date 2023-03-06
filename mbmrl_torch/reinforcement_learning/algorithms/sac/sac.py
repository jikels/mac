import itertools
import numpy as np
import torch
import time
import cherry as ch
import wandb
from copy import deepcopy
from torch.optim import Adam
from mbmrl_torch.gym.utils.env_init import init_env

'''
The code was adapted from: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
'''

class Sac():

    def __init__(self,config=None,model_handler=None):

        if config is not None:
            env_name=config["env_name"]
            action_dim=config["action_dim"]
            save_dir = config["model_data_dir"] + "/" + \
            config["env_name"]
            seed=config["seed"]
            replay_size=config["replay_size"]
            # ent_coef= 'auto',
            gamma=config["gamma"]
            polyak=config["polyak"]
            lr=config["lr"]
            alpha=config["alpha"]
            batch_size=config["batch_size"]
            start_steps=config["start_steps"]
            update_after=config["update_after"]
            update_every=config["update_every"]
            num_test_episodes=config["num_test_episodes"]
            max_episode_len=config["max_episode_len"]
            save_freq=config["save_freq"]
            alpha_lr=config["alpha_lr"]
            gradient_steps=config["gradient_steps"]
            obs_normalization = config["obs_normalization"]
            rew_normalization = config["rew_normalization"]
            num_envs = config["num_envs"]
            total_steps = config["total_steps"]
        else:
            # init model handler and create model
            quit()
            #model_handler = ModelHandlerAC(self.config,self.path_model_data,self.device)
            #model_handler.create_model()
        
        self.model_handler = model_handler
        self.seed = seed
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = start_steps #simplified by omitting update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_episode_len = max_episode_len
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.gradient_steps = update_every #simplified by omitting gradient_steps
        self.obs_normalization = obs_normalization
        self.rew_normalization = rew_normalization
        self.total_steps = total_steps
        self.alpha_lr= lr #simplified by omitting alpha_lr
        
        torch.set_num_threads(torch.get_num_threads())

        self.num_envs = num_envs

        def env_fn_train():
            env = init_env(
                env_name=env_name,
                num_envs=self.num_envs
            )
            return env

        def env_fn_test():
            env = init_env(
                env_name=env_name,
                num_envs=self.num_test_episodes
            )
            return env

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
        self.env, self.test_env = env_fn_train(), env_fn_test()

        # Set actor-critic module and target networks
        self.ac = self.model_handler.model
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via
        # polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ch.ExperienceReplay()

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # set up entropy optimizer
        if self.alpha == 'auto':
            self.alpha = 1.0
            self.automatic_entropy = True
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1).cuda()
            self.log_alpha.requires_grad = True
            self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.automatic_entropy = False

    def train(self):
        # interaction params
        # ep = epoch NOT episode

        # general
        _total_steps = int(self.total_steps)//self.num_envs
        self.epochs = _total_steps // self.max_episode_len
        start_time = time.time()

        # env general
        o = self.env.reset()
        ep_ret = np.zeros(self.num_envs)

        # env normalization
        mean_std_obs = torch.zeros(2).cuda()
        std_rew = 0 

        # env vector-env
        episode_lens = np.zeros(self.num_envs)
        episode_lens_sum = 0
        num_collected_episodes = 0
        num_collected_episodes_ep = 0

        # losses
        q_loss = 0
        pi_loss = 0
        pi_info = 0

        # log helpers
        total_mean_ep_test_reward = 0
        total_mean_ep_train_reward = 0
        
        # Main loop: collect experience in env and update/log each epoch
        for t in range(_total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > (self.start_steps//self.num_envs):
                a = self.get_action(o)
            else:
                a = self.sample_action()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            episode_lens +=1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            if np.all(episode_lens!=self.max_episode_len)==False:
                d = np.where(d==True, False, d)
                for idx,len in enumerate(episode_lens):
                    if len==self.max_episode_len:
                        num_collected_episodes += 1
                        num_collected_episodes_ep += 1
                        episode_lens_sum += episode_lens[idx]
                        episode_lens[idx] = 0
                        # o = self.env.reset() -> automatically by vec env
            
            # count closed episodes of vector env
            if np.all(d==False)==False:
                for idx,done in enumerate(d):
                    if done:
                        num_collected_episodes += 1
                        num_collected_episodes_ep += 1
                        episode_lens_sum += episode_lens[idx]
                        episode_lens[idx] = 0 
                
            # Store experience to replay buffer
            self.add_to_replay(o,a,o2,r,d)

            # update most recent observation
            o = o2

            # Update handling
            if t >= (self.update_after//self.num_envs) and t%(self.update_every//self.num_envs)==0:
                
                # update observation stats
                if self.obs_normalization:
                    mean_std_obs = self.env.get_observation_stats_torch()
                    self.ac.set_stats(mean_std_obs)
                    self.ac_targ.set_stats(mean_std_obs)

                # train actor critic
                # draft (not tested): batch x times and push data to GPU!
                # data_len = int(self.batch_size*self.gradient_steps)
                # data = self.replay_buffer.sample(data_len)
                # data = data.to('cuda:0')
                for i in range(self.gradient_steps):
                    # start_index = int(i*self.batch_size)
                    # end_index = int(data_len) if i == self.gradient_steps - \
                        # 1 else int(i*self.batch_size + self.batch_size)
                    data = self.replay_buffer.sample(self.batch_size)
                    q_loss, pi_loss, q_info, pi_info, std_rew = self.update(data=data)

            # End of epoch handling
            if self.time_to_log(t,self.max_episode_len,1):
                epoch = (t + 1) // self.max_episode_len
                
                # Save model
                if self.save_freq is not None:
                    if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                        self.model_handler.save_model()

                # update train_return stats
                mean_ep_train_reward = np.mean(ep_ret)
                total_mean_ep_train_reward += mean_ep_train_reward
                running_mean_ep_train_reward = total_mean_ep_train_reward/epoch

                # Test the performance of the deterministic version of the
                # agent.
                mean_ep_test_reward, test_ep_dones = self.test_agent()

                # update test_return stats
                total_mean_ep_test_reward += mean_ep_test_reward
                running_mean_ep_test_reward = total_mean_ep_test_reward/epoch

                # track epoch time
                ep_time = time.time() - start_time

                dict = {
                    'epoch': epoch,
                    'episodes_collected': num_collected_episodes,
                    'collected_episodes_epoch': num_collected_episodes_ep,
                    'average_episode_length': round(episode_lens_sum/num_collected_episodes_ep,2),
                    'mean_train_rew_epoch': mean_ep_train_reward,
                    'running_mean_train_rew_epoch': running_mean_ep_train_reward,
                    'mean_test_rew_episode': mean_ep_test_reward,
                    'running_mean_test_rew_episode': running_mean_ep_test_reward,
                    'test_episode_dones': test_ep_dones,
                    'total_env_interact': t*self.num_envs,
                    'entropy_coefficient': self.alpha,
                    'mean_obs': wandb.Histogram(mean_std_obs[0].detach().cpu().numpy()),
                    'std_obs': wandb.Histogram(mean_std_obs[1].detach().cpu().numpy()),
                    'std_rew': std_rew,
                    'LogPi': pi_info,
                    'Loss_Pi': pi_loss,
                    'Loss_Q': q_loss,
                    'Time_per_Epoch': ep_time,
                    'Estimated_Time_to_completion_(min)': round(((self.epochs-epoch)*ep_time)/60,2)}

                # reset params
                ep_ret = np.zeros(self.num_envs)
                episode_lens_sum = 0
                num_collected_episodes_ep = 0

                # log
                wandb.log(dict)
                start_time = time.time()
    
    def time_to_log(self,s,s_ep,log_int):
        step = s+1
        log = s_ep * log_int
        return step%log==0

    def update(self, data):
        o, a, r, o2, d = data.state().cuda(), data.action(
        ).cuda(), data.reward().cuda(), data.next_state().cuda(), data.done().cuda()

        if self.rew_normalization:
            # normalize reward
            std_rew = self.env.get_reward_std()
            r = r / std_rew
        else:
            std_rew = np.zeros(1)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(o, a, r, o2, d)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, logp_pi, pi_info = self.compute_loss_pi(o)
        loss_pi.backward()
        self.pi_optimizer.step()

        # update the entropy coefficient
        if self.automatic_entropy:
            self.alpha = self.log_alpha.exp()
            alpha_loss = -(self.log_alpha * (self.target_entropy+logp_pi).detach()) #Karam: detach() hinter log_pi
            alpha_loss = alpha_loss.mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new
                # tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q.item(), loss_pi.item(), q_info, pi_info, std_rew

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).cuda(),
                           deterministic)

    def test_agent(self):
        o = self.test_env.reset()
        ep_ret= np.zeros(self.num_test_episodes)
        dones = np.zeros(self.num_test_episodes)
        ep_len = 0
        while not(ep_len == self.max_episode_len):
            # Take deterministic actions at test time
            o, r, d, _ = self.test_env.step(self.get_action(o, True))

            ep_ret += r
            ep_len += 1

            # ignore done signal if max steps
            if ep_len==self.max_episode_len:
                d = np.where(d==True, False, d)

            # Log if done state has been reached
            if np.all(d==False)==False:
                for idx, done in enumerate(d):
                    if done:
                        dones[idx] += 1

        return np.mean(ep_ret), np.sum(dones)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, o, a, r, o2, d):

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * \
                (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, o):
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = logp_pi.cpu().detach().numpy()

        return loss_pi, logp_pi, pi_info

    def count_vars(module):
        return sum([np.prod(p.shape) for p in module.parameters()])
    
    def sample_action(self):
        action = []
        for i in range(len(self.env.action_space)):
            a = self.env.action_space[i].sample()
            action.append(a)
        return np.array(action)

    def add_to_replay(self,o1,action,o2,r,done):
        for i in range(len(o2)):
            self.replay_buffer.append(o1[i],action[i],r[i],o2[i],done[i])

    