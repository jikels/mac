import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.env_stats=None

    def forward(self, obs, deterministic=False, with_logprob=True):

        if self.env_stats is not None:
            obs = (obs-self.env_stats[0])/self.env_stats[1]

        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action -
                        F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)
        self.env_stats = None

    def forward(self, obs, act):

        if self.env_stats is not None:
            obs = (obs-self.env_stats[0])/self.env_stats[1]

        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        hidden_sizes=(256, 256),
        activation=nn.ReLU):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        super().__init__()

        # build policy
        self.pi = SquashedGaussianMLPActor(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            self.activation,
            self.act_limit)
        
        #build value function q1
        self.q1 = MLPQFunction(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            self.activation)

        # build value function q2
        self.q2 = MLPQFunction(
            self.obs_dim,
            self.act_dim,
            self.hidden_sizes,
            self.activation)

    def set_stats(self, env_stats):
        self.pi.env_stats = env_stats
        self.q1.env_stats = env_stats
        self.q2.env_stats = env_stats

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().squeeze().numpy()
    
    def act_tensor(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a
    
    def save(self, file_path):
        kwargs = {  "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim, 
                    "act_limit": self.act_limit,  
                    "hidden_sizes": self.hidden_sizes,
                    "activation": self.activation}
        state_dict = self.state_dict()
        others = {  "env_stats_pi": self.pi.env_stats,
                    "env_stats_q1": self.q1.env_stats,
                    "env_stats_q2": self.q2.env_stats}
        torch.save({"kwargs":kwargs, "state_dict":state_dict, "others":others}, file_path)