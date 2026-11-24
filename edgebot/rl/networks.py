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


class RunningNorm(nn.Module):
    """Running mean/variance observation normaliser (SimBa component 1).

    Statistics change only when update() is called (during rollout collection);
    they are saved with the model and frozen at evaluation and export."""

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


class SimBaBlock(nn.Module):
    """Pre-LayerNorm residual feed-forward block (SimBa component 2)."""

    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 4 * h), nn.ReLU(), nn.Linear(4 * h, h))

    def forward(self, x):
        return x + self.net(x)


class SimBa(nn.Module):
    """SimBa network (Lee et al., ICLR 2025): observation normalisation, residual blocks and a
    final LayerNorm (component 3). Biases the network towards simple, well-generalising functions."""

    def __init__(self, norm: RunningNorm, in_dim: int, out_dim: int, hidden: int = 128, blocks: int = 2,
                 out_std: float = 1.0):
        super().__init__()
        self.norm = norm
        self.inp = layer_init(nn.Linear(in_dim, hidden))
        self.blocks = nn.Sequential(*[SimBaBlock(hidden) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.out = layer_init(nn.Linear(hidden, out_dim), std=out_std)

    def forward(self, x):
        return self.out(self.ln(self.blocks(self.inp(self.norm(x)))))


class ActorCritic(nn.Module):
    """Separate actor and critic networks; Gaussian (continuous) or categorical (discrete) policy.

    arch="mlp" is the plain tanh MLP used by the deployed policy; arch="simba" uses SimBa networks."""

    def __init__(self, obs_dim: int, act_dim: int, continuous: bool = True, hidden=(128, 128), arch: str = "mlp"):
        super().__init__()
        self.continuous = continuous
        self.obs_norm = None
        if arch == "simba":
            self.obs_norm = RunningNorm(obs_dim)
            self.critic = SimBa(self.obs_norm, obs_dim, 1, hidden[0], len(hidden), out_std=1.0)
            self.actor = SimBa(self.obs_norm, obs_dim, act_dim, hidden[0], len(hidden), out_std=0.01)
        else:
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
