"""JointBERT with a pretrained transformer backbone (DistilBERT by default).

Fine-tuning a pretrained encoder transfers lexical knowledge learnt from large
corpora, which is what lets the model understand verbs it never saw in the robot
command data. Word-level BIO labels are aligned to the first WordPiece of each
word; the remaining pieces are ignored in the loss. The fine-tuned model also acts
as the teacher for knowledge distillation into the edge-sized JointBERT-mini.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgebot.nlu.data import INTENTS, TAGS
from edgebot.nlu.engine import INTENT2ID, TAG2ID, confusion, evaluate_predictions


class JointBERT(nn.Module):
    def __init__(self, backbone: str = "distilbert-base-uncased"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(backbone)
        self.enc = AutoModel.from_pretrained(backbone)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.intent_head = nn.Linear(h, len(INTENTS))
        self.slot_head = nn.Linear(h, len(TAGS))

    def encode(self, token_lists, device):
        enc = self.tok(token_lists, is_split_into_words=True, padding=True, truncation=True, max_length=64,
                       return_tensors="pt")
        word_ids = [enc.word_ids(i) for i in range(len(token_lists))]
        return {k: v.to(device) for k, v in enc.items()}, word_ids

    def forward(self, enc):
        h = self.enc(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        return self.intent_head(self.drop(h[:, 0])), self.slot_head(self.drop(h))

    @staticmethod
    def first_piece_index(word_ids, n_words):
        idx, seen = [0] * n_words, set()
        for j, w in enumerate(word_ids):
            if w is not None and w not in seen:
                seen.add(w)
                idx[w] = j
        return idx

    @torch.no_grad()
    def predict(self, token_lists, device, bs: int = 128):
        self.eval()
        intents, tags, intent_probs, slot_probs = [], [], [], []
        for i in range(0, len(token_lists), bs):
            chunk = token_lists[i:i + bs]
            enc, wids = self.encode(chunk, device)
            il, sl = self(enc)
            ip = F.softmax(il, -1).cpu()
            sp = F.softmax(sl, -1).cpu()
            for k, toks in enumerate(chunk):
                first = self.first_piece_index(wids[k], len(toks))
                p_words = sp[k, first]
                intents.append(INTENTS[int(ip[k].argmax())])
                tags.append([TAGS[int(t)] for t in p_words.argmax(-1)])
                intent_probs.append(ip[k].numpy())
                slot_probs.append(p_words.numpy())
        return intents, tags, intent_probs, slot_probs


def train_joint_bert(splits: dict, epochs: int = 4, lr: float = 5e-5, bs: int = 32, device="cuda",
                     backbone: str = "distilbert-base-uncased", seed: int = 7, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = JointBERT(backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train = splits["train"]
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train))
        tot = 0.0
        for i in range(0, len(train), bs):
            batch = [train[j] for j in perm[i:i + bs]]
            enc, wids = model.encode([e.tokens for e in batch], device)
            labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
            for k, e in enumerate(batch):
                first = model.first_piece_index(wids[k], len(e.tokens))
                for w, j in enumerate(first):
                    labels[k, j] = TAG2ID[e.tags[w]]
            yi = torch.tensor([INTENT2ID[e.intent] for e in batch], device=device)
            il, sl = model(enc)
            loss = F.cross_entropy(il, yi) + F.cross_entropy(sl.reshape(-1, len(TAGS)), labels.to(device).reshape(-1),
                                                             ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
        log(f"[JointBERT] epoch {ep} | loss {tot / len(train):.4f} | {time.time() - t0:.0f}s")
    results = {"backbone": backbone, "n_params": sum(p.numel() for p in model.parameters()),
               "train_time_s": time.time() - t0}
    for name in ("test", "test_noisy"):
        data = splits[name]
        pi, pt, _, _ = model.predict([e.tokens for e in data], device)
        m = evaluate_predictions(data, pi, pt)
        m["confusion"] = confusion(data, pi).tolist()
        results[name] = m
    # latency of a single utterance (batch 1), the deployment setting
    toks = ["move", "the", "red", "box", "from", "shelf", "a", "to", "the", "conveyor"]
    for dev in ("cpu", device):
        model.to(dev)
        lat = []
        for _ in range(30):
            t = time.perf_counter()
            model.predict([toks], dev)
            lat.append(time.perf_counter() - t)
        results[f"latency_ms_{dev}"] = float(np.median(lat[5:]) * 1000)
    model.to(device)
    return model, results
