"""Neural network function approximators used by the deep RL agents."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim: int, out_dim: int, hidden=(128, 128), act=nn.Tanh, out_std: float = 1.0) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(d, h)), act()]
        d = h
    layers.append(layer_init(nn.Linear(d, out_dim), std=out_std))
    return nn.Sequential(*layers)


class NatureCNN(nn.Module):
    """DQN-style convolutional encoder (Mnih et al., 2015) sized for 64x64 inputs."""

    def __init__(self, in_ch: int, feat: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_ch, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, feat)), nn.ReLU(),
        )
        self.out_dim = feat

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs; Gaussian (continuous) or categorical (discrete) policy."""

    def __init__(self, obs_dim: int, act_dim: int, continuous: bool = True, hidden=(128, 128)):
        super().__init__()
        self.continuous = continuous
        self.critic = mlp(obs_dim, 1, hidden, out_std=1.0)
        self.actor = mlp(obs_dim, act_dim, hidden, out_std=0.01)
        if continuous:
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def dist(self, obs):
        logits = self.actor(obs)
        if self.continuous:
            return Normal(logits, self.log_std.exp().expand_as(logits))
        return Categorical(logits=logits)

    def act(self, obs, deterministic: bool = False):
        d = self.dist(obs)
        if deterministic:
            a = d.mean if self.continuous else d.probs.argmax(-1)
        else:
            a = d.sample()
        logp = d.log_prob(a).sum(-1) if self.continuous else d.log_prob(a)
        return a, logp, self.value(obs)

    def evaluate(self, obs, actions):
        d = self.dist(obs)
        if self.continuous:
            return d.log_prob(actions).sum(-1), d.entropy().sum(-1), self.value(obs)
        return d.log_prob(actions.long()), d.entropy(), self.value(obs)


class DeterministicActor(nn.Module):
    """Deployment wrapper: maps an observation to the mean action (exported to ONNX)."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.actor = ac.actor
        self.continuous = ac.continuous

    def forward(self, obs):
        out = self.actor(obs)
        return torch.clamp(out, -1.0, 1.0) if self.continuous else out


class QNetwork(nn.Module):
    """Q(s, .) head over an MLP or CNN torso, optionally with a dueling decomposition."""

    def __init__(self, obs_shape, n_actions: int, dueling: bool = True, hidden: int = 256):
        super().__init__()
        self.pixels = len(obs_shape) == 3
        if self.pixels:
            self.torso = NatureCNN(obs_shape[0], hidden)
            feat = self.torso.out_dim
        else:
            self.torso = nn.Sequential(layer_init(nn.Linear(obs_shape[0], hidden)), nn.ReLU(),
                                       layer_init(nn.Linear(hidden, hidden)), nn.ReLU())
            feat = hidden
        self.dueling = dueling
        if dueling:
            self.value = layer_init(nn.Linear(feat, 1), std=1.0)
            self.adv = layer_init(nn.Linear(feat, n_actions), std=0.01)
        else:
            self.q = layer_init(nn.Linear(feat, n_actions), std=0.01)

    def forward(self, x):
        h = self.torso(x)
        if self.dueling:
            a = self.adv(h)
            return self.value(h) + a - a.mean(dim=-1, keepdim=True)
        return self.q(h)
