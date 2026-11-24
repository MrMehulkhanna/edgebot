"""Proximal Policy Optimisation (Schulman et al., 2017) with GAE(lambda).

Supports continuous (diagonal Gaussian) and discrete action spaces and runs on a
synchronous vector of environments. Truncated episodes are bootstrapped with
V(s_final) so time limits do not bias the value targets.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from edgebot.rl.networks import ActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 16
    n_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    epochs: int = 10
    minibatch: int = 1024
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden: tuple = (128, 128)
    seed: int = 0


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, continuous: bool, cfg: PPOConfig, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.continuous = continuous
        self.ac = ActorCritic(obs_dim, act_dim, continuous, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        single = o.dim() == 1
        if single:
            o = o[None]
        a, _, _ = self.ac.act(o, deterministic)
        a = a.cpu().numpy()
        return a[0] if single else a

    def train(self, venv, log_every: int = 10, callback=None) -> dict:
        cfg, dev = self.cfg, self.device
        n, T = cfg.n_envs, cfg.n_steps
        obs_shape = venv.observation_space.shape
        act_shape = venv.action_space.shape if self.continuous else ()

        buf_obs = torch.zeros((T, n) + obs_shape, device=dev)
        buf_act = torch.zeros((T, n) + act_shape, device=dev)
        buf_logp = torch.zeros((T, n), device=dev)
        buf_rew = torch.zeros((T, n), device=dev)
        buf_done = torch.zeros((T, n), device=dev)
        buf_val = torch.zeros((T, n), device=dev)

        obs = torch.as_tensor(venv.reset(), dtype=torch.float32, device=dev)
        ep_ret = np.zeros(n)
        ep_len = np.zeros(n, dtype=int)
        recent = deque(maxlen=100)
        history = {"steps": [], "return": [], "success": [], "collision": [], "ep_len": [],
                   "kl": [], "entropy": [], "time": []}
        n_updates = cfg.total_steps // (n * T)
        t0 = time.time()
        global_step = 0

        for update in range(1, n_updates + 1):
            frac = 1.0 - (update - 1) / n_updates
            for g in self.opt.param_groups:
                g["lr"] = frac * cfg.lr

            for t in range(T):
                with torch.no_grad():
                    a, logp, v = self.ac.act(obs)
                a_np = a.cpu().numpy()
                next_obs, raw_rew, done, infos = venv.step(a_np)
                rew = raw_rew.copy()
                for i, info in enumerate(infos):
                    if info.get("truncated") and not info.get("terminated"):
                        with torch.no_grad():
                            fo = torch.as_tensor(info["final_obs"], dtype=torch.float32, device=dev)[None]
                            rew[i] += cfg.gamma * self.ac.value(fo).item()
                buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = obs, a, logp, v
                buf_rew[t] = torch.as_tensor(rew, device=dev)
                buf_done[t] = torch.as_tensor(done, dtype=torch.float32, device=dev)
                ep_ret += raw_rew
                ep_len += 1
                for i, d in enumerate(done):
                    if d:
                        recent.append((ep_ret[i], float(infos[i].get("success", 0)),
                                       float(infos[i].get("collision", 0)), ep_len[i]))
                        ep_ret[i], ep_len[i] = 0, 0
                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
                global_step += n

            # ---- GAE
            with torch.no_grad():
                next_v = self.ac.value(obs)
                adv = torch.zeros_like(buf_rew)
                last = torch.zeros(n, device=dev)
                for t in reversed(range(T)):
                    nv = next_v if t == T - 1 else buf_val[t + 1]
                    nonterm = 1.0 - buf_done[t]
                    delta = buf_rew[t] + cfg.gamma * nv * nonterm - buf_val[t]
                    last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                    adv[t] = last
                ret = adv + buf_val

            b_obs = buf_obs.reshape((-1,) + obs_shape)
            b_act = buf_act.reshape((-1,) + act_shape)
            b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
            idx = np.arange(n * T)
            kls, ents = [], []
            for _ in range(cfg.epochs):
                np.random.shuffle(idx)
                stop = False
                for s in range(0, len(idx), cfg.minibatch):
                    mb = idx[s:s + cfg.minibatch]
                    new_logp, ent, v = self.ac.evaluate(b_obs[mb], b_act[mb])
                    ratio = (new_logp - b_logp[mb]).exp()
                    mb_adv = b_adv[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    v_loss = 0.5 * (v - b_ret[mb]).pow(2).mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean()
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                    self.opt.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (new_logp - b_logp[mb])).mean().item()
                    kls.append(approx_kl)
                    ents.append(ent.mean().item())
                    if approx_kl > cfg.target_kl:
                        stop = True
                        break
                if stop:
                    break

            if recent and (update % log_every == 0 or update == n_updates):
                arr = np.array(recent)
                history["steps"].append(global_step)
                history["return"].append(float(arr[:, 0].mean()))
                history["success"].append(float(arr[:, 1].mean()))
                history["collision"].append(float(arr[:, 2].mean()))
                history["ep_len"].append(float(arr[:, 3].mean()))
                history["kl"].append(float(np.mean(kls)))
                history["entropy"].append(float(np.mean(ents)))
                history["time"].append(time.time() - t0)
                print(f"[PPO] step {global_step:>8d} | return {arr[:, 0].mean():7.2f} | "
                      f"success {arr[:, 1].mean():.2f} | collision {arr[:, 2].mean():.2f} | "
                      f"{global_step / (time.time() - t0):.0f} sps", flush=True)
                if callback is not None:
                    callback(self, history)
        return history

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"state_dict": self.ac.state_dict(), "cfg": asdict(self.cfg),
                    "continuous": self.continuous, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, obs_dim: int, act_dim: int, device="cpu") -> "PPO":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = PPOConfig(**{**ck["cfg"], "hidden": tuple(ck["cfg"]["hidden"])})
        agent = cls(obs_dim, act_dim, ck["continuous"], cfg, device)
        agent.ac.load_state_dict(ck["state_dict"])
        agent.ac.eval()
        return agent
