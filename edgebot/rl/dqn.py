"""Deep Q-Network (Mnih et al., 2015) with Double-DQN targets and a dueling head.

Works with vector (lidar) and image (pixels) observations. Image transitions are
stored as uint8 to keep the replay memory small enough for a laptop.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from edgebot.rl.networks import QNetwork


@dataclass
class DQNConfig:
    total_steps: int = 400_000
    n_envs: int = 8
    buffer_size: int = 200_000
    batch_size: int = 256
    lr: float = 5e-4
    gamma: float = 0.99
    learning_starts: int = 10_000
    train_freq: int = 1          # vector steps between updates
    gradient_steps: int = 2
    target_update: int = 2_000   # gradient steps between hard target syncs
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_fraction: float = 0.3
    double: bool = True
    dueling: bool = True
    hidden: int = 256
    seed: int = 0


class ReplayBuffer:
    def __init__(self, size: int, obs_shape, pixels: bool):
        self.size, self.ptr, self.full = size, 0, False
        dt = np.uint8 if pixels else np.float32
        self.scale = 255.0 if pixels else 1.0
        self.obs = np.zeros((size,) + obs_shape, dtype=dt)
        self.next_obs = np.zeros((size,) + obs_shape, dtype=dt)
        self.act = np.zeros(size, dtype=np.int64)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)

    def _enc(self, o):
        return (o * 255).astype(np.uint8) if self.scale > 1 else o

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i], self.next_obs[i] = self._enc(o), self._enc(no)
        self.act[i], self.rew[i], self.done[i] = a, r, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n: int, device):
        idx = np.random.randint(0, len(self), n)
        f = lambda x: torch.as_tensor(x[idx], device=device)
        o, no = f(self.obs).float() / self.scale, f(self.next_obs).float() / self.scale
        return o, f(self.act), f(self.rew), no, f(self.done)


class DQN:
    def __init__(self, obs_shape, n_actions: int, cfg: DQNConfig, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_shape = tuple(obs_shape)
        self.n_actions = n_actions
        self.q = QNetwork(self.obs_shape, n_actions, cfg.dueling, cfg.hidden).to(self.device)
        self.q_target = QNetwork(self.obs_shape, n_actions, cfg.dueling, cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, eps: float = 0.0):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        single = o.dim() == len(self.obs_shape)
        if single:
            o = o[None]
        a = self.q(o).argmax(-1).cpu().numpy()
        if eps > 0:
            rand = np.random.rand(len(a)) < eps
            a[rand] = np.random.randint(0, self.n_actions, rand.sum())
        return int(a[0]) if single else a

    def _update(self, buf: ReplayBuffer) -> float:
        cfg = self.cfg
        o, a, r, no, d = buf.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            if cfg.double:
                na = self.q(no).argmax(-1, keepdim=True)
                nq = self.q_target(no).gather(1, na).squeeze(1)
            else:
                nq = self.q_target(no).max(-1).values
            target = r + cfg.gamma * (1 - d) * nq
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def train(self, venv, log_every_steps: int = 10_000) -> dict:
        cfg = self.cfg
        pixels = len(self.obs_shape) == 3
        buf = ReplayBuffer(cfg.buffer_size, self.obs_shape, pixels)
        obs = venv.reset()
        n = venv.n
        ep_ret, ep_len = np.zeros(n), np.zeros(n, dtype=int)
        recent = deque(maxlen=100)
        history = {"steps": [], "return": [], "success": [], "collision": [], "ep_len": [],
                   "eps": [], "loss": [], "time": []}
        losses = deque(maxlen=500)
        n_grad, step, next_log = 0, 0, log_every_steps
        t0 = time.time()
        while step < cfg.total_steps:
            eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step /
                      (cfg.eps_fraction * cfg.total_steps))
            act = self.predict(obs, eps)
            next_obs, rew, done, infos = venv.step(act)
            for i in range(n):
                term = infos[i]["terminated"]
                real_next = infos[i]["final_obs"] if done[i] else next_obs[i]
                buf.add(obs[i], act[i], rew[i], real_next, float(term))
                ep_ret[i] += rew[i]
                ep_len[i] += 1
                if done[i]:
                    recent.append((ep_ret[i], float(infos[i]["success"]), float(infos[i]["collision"]), ep_len[i]))
                    ep_ret[i], ep_len[i] = 0, 0
            obs = next_obs
            step += n
            if step >= cfg.learning_starts and (step // n) % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    losses.append(self._update(buf))
                    n_grad += 1
                    if n_grad % cfg.target_update == 0:
                        self.q_target.load_state_dict(self.q.state_dict())
            if step >= next_log and recent:
                next_log += log_every_steps
                arr = np.array(recent)
                history["steps"].append(step)
                history["return"].append(float(arr[:, 0].mean()))
                history["success"].append(float(arr[:, 1].mean()))
                history["collision"].append(float(arr[:, 2].mean()))
                history["ep_len"].append(float(arr[:, 3].mean()))
                history["eps"].append(eps)
                history["loss"].append(float(np.mean(losses)) if losses else 0.0)
                history["time"].append(time.time() - t0)
                print(f"[DQN] step {step:>8d} | return {arr[:, 0].mean():7.2f} | success {arr[:, 1].mean():.2f}"
                      f" | collision {arr[:, 2].mean():.2f} | eps {eps:.2f} | {step / (time.time() - t0):.0f} sps",
                      flush=True)
        return history

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"state_dict": self.q.state_dict(), "cfg": asdict(self.cfg), "obs_shape": self.obs_shape,
                    "n_actions": self.n_actions, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, device="cpu") -> "DQN":
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ck["obs_shape"], ck["n_actions"], DQNConfig(**ck["cfg"]), device)
        agent.q.load_state_dict(ck["state_dict"])
        agent.q.eval()
        return agent
