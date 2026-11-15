"""Training, evaluation and inference for the joint intent / slot model, plus the
conversion from BIO tags to a structured command frame."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgebot.nlu.data import INTENTS, TAGS, Example, build_splits, canonical, tokenize
from edgebot.nlu.model import BoWIntentClassifier, JointBERTMini, Vocab, char_ids

INTENT2ID = {k: i for i, k in enumerate(INTENTS)}
TAG2ID = {k: i for i, k in enumerate(TAGS)}


# ---------------------------------------------------------------- batching
def encode_batch(token_lists: list[list[str]], vocab: Vocab, device="cpu", word_dropout: float = 0.0):
    T = max(len(t) for t in token_lists)
    words = torch.zeros(len(token_lists), T, dtype=torch.long)
    chars = torch.zeros(len(token_lists), T, 16, dtype=torch.long)
    mask = torch.zeros(len(token_lists), T, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        ids = vocab.encode(toks)
        if word_dropout > 0:
            ids = [1 if np.random.rand() < word_dropout else x for x in ids]
        words[i, :len(ids)] = torch.tensor(ids)
        chars[i, :len(ids)] = torch.tensor(char_ids(toks))
        mask[i, :len(ids)] = 1
    return words.to(device), chars.to(device), mask.to(device)


def _targets(batch: list[Example], device):
    T = max(len(e.tokens) for e in batch)
    tags = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        tags[i, :len(e.tags)] = torch.tensor([TAG2ID[t] for t in e.tags])
    intents = torch.tensor([INTENT2ID[e.intent] for e in batch])
    return intents.to(device), tags.to(device)


# ---------------------------------------------------------------- metrics
def spans(tags: list[str]) -> set:
    out, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and t[2:] != typ):
            if typ is not None:
                out.add((start, i, typ))
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return out


def evaluate_predictions(gold: list[Example], pred_intents: list[str], pred_tags: list[list[str]]) -> dict:
    tp = fp = fn = 0
    exact = 0
    for e, pi, pt in zip(gold, pred_intents, pred_tags):
        g, p = spans(e.tags), spans(pt)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        exact += int(pi == e.intent and g == p)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "intent_acc": float(np.mean([pi == e.intent for e, pi in zip(gold, pred_intents)])),
        "slot_precision": prec, "slot_recall": rec,
        "slot_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "frame_acc": exact / len(gold),
    }


def confusion(gold: list[Example], pred_intents: list[str]) -> np.ndarray:
    M = np.zeros((len(INTENTS), len(INTENTS)), dtype=int)
    for e, p in zip(gold, pred_intents):
        M[INTENT2ID[e.intent], INTENT2ID[p]] += 1
    return M


# ---------------------------------------------------------------- frames
@dataclass
class Frame:
    intent: str
    confidence: float
    slots: dict = field(default_factory=dict)
    tokens: list = field(default_factory=list)
    tags: list = field(default_factory=list)

    def to_dict(self):
        return {"intent": self.intent, "confidence": round(self.confidence, 4), "slots": self.slots}


def tags_to_slots(tokens: list[str], tags: list[str]) -> dict:
    slots: dict = {}
    for s, e, typ in sorted(spans(tags)):
        value = " ".join(canonical(w) for w in tokens[s:e])
        if typ == "color":
            slots.setdefault("colors", []).append(value)
        else:
            slots[typ] = value
    if "colors" in slots and len(slots["colors"]) == 1:
        slots["color"] = slots.pop("colors")[0]
    return slots


# ---------------------------------------------------------------- model wrapper
class NLU:
    def __init__(self, model: JointBERTMini, vocab: Vocab, device="cpu"):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = torch.device(device)

    @torch.no_grad()
    def predict_tokens(self, token_lists: list[list[str]]):
        words, chars, mask = encode_batch(token_lists, self.vocab, self.device)
        il, sl = self.model(words, chars, mask)
        probs = F.softmax(il, -1)
        conf, ii = probs.max(-1)
        st = sl.argmax(-1).cpu().numpy()
        intents = [INTENTS[i] for i in ii.cpu().numpy()]
        tags = [[TAGS[t] for t in st[k, :len(toks)]] for k, toks in enumerate(token_lists)]
        return intents, tags, conf.cpu().numpy()

    def parse(self, text: str) -> Frame:
        toks = tokenize(text)
        if not toks:
            return Frame("unknown", 0.0)
        intents, tags, conf = self.predict_tokens([toks])
        return Frame(intents[0], float(conf[0]), tags_to_slots(toks, tags[0]), toks, tags[0])

    def evaluate(self, data: list[Example], bs: int = 256) -> dict:
        pi, pt = [], []
        for i in range(0, len(data), bs):
            a, b, _ = self.predict_tokens([e.tokens for e in data[i:i + bs]])
            pi += a
            pt += b
        m = evaluate_predictions(data, pi, pt)
        m["confusion"] = confusion(data, pi).tolist()
        return m

    def save(self, path: str, meta: dict | None = None):
        torch.save({"state_dict": self.model.state_dict(), "itos": self.vocab.itos,
                    "config": self.model_config, "meta": meta or {}}, path)

    @property
    def model_config(self):
        m = self.model
        return {"vocab_size": m.word.num_embeddings, "n_intents": m.intent_head[-1].out_features,
                "n_tags": m.slot_head.out_features, "d_model": m.d_model,
                "n_layers": len(m.encoder.layers)}

    @classmethod
    def load(cls, path: str, device="cpu") -> "NLU":
        ck = torch.load(path, map_location=device, weights_only=False)
        vocab = Vocab([])
        vocab.itos = ck["itos"]
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        cfg = ck["config"]
        model = JointBERTMini(cfg["vocab_size"], cfg["n_intents"], cfg["n_tags"], cfg["d_model"],
                              n_layers=cfg["n_layers"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, vocab, device)


def train_nlu(epochs: int = 30, seed: int = 7, device="cpu", lr: float = 2e-3, bs: int = 64,
              d_model: int = 128, n_layers: int = 2, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = build_splits(seed)
    vocab = Vocab([w for e in splits["train"] for w in e.tokens], min_freq=2)
    model = JointBERTMini(len(vocab), len(INTENTS), len(TAGS), d_model, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train = splits["train"]
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history = {"epoch": [], "loss": [], "val_intent_acc": [], "val_slot_f1": [], "time": []}
    nlu = NLU(model, vocab, device)
    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train))
        tot = 0.0
        for i in range(0, len(train), bs):
            batch = [train[j] for j in perm[i:i + bs]]
            words, chars, mask = encode_batch([e.tokens for e in batch], vocab, device, word_dropout=0.1)
            yi, yt = _targets(batch, device)
            il, sl = model(words, chars, mask)
            loss = F.cross_entropy(il, yi, label_smoothing=0.05) + \
                F.cross_entropy(sl.reshape(-1, sl.shape[-1]), yt.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
        model.eval()
        val = nlu.evaluate(splits["val"])
        score = val["intent_acc"] + val["slot_f1"]
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history["epoch"].append(ep)
        history["loss"].append(tot / len(train))
        history["val_intent_acc"].append(val["intent_acc"])
        history["val_slot_f1"].append(val["slot_f1"])
        history["time"].append(time.time() - t0)
        log(f"[NLU] epoch {ep:2d} | loss {tot / len(train):.4f} | val intent {val['intent_acc']:.4f} "
            f"| val slot-F1 {val['slot_f1']:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return nlu, splits, history


def train_bow_baseline(splits: dict, epochs: int = 40, seed: int = 7) -> dict:
    """Bag-of-Words logistic regression for intent classification only."""
    torch.manual_seed(seed)
    vocab = Vocab([w for e in splits["train"] for w in e.tokens])

    def featurise(data):
        X = torch.zeros(len(data), len(vocab))
        for i, e in enumerate(data):
            for w in vocab.encode(e.tokens):
                X[i, w] += 1.0
        return X, torch.tensor([INTENT2ID[e.intent] for e in data])

    Xtr, ytr = featurise(splits["train"])
    model = BoWIntentClassifier(len(vocab), len(INTENTS))
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(model(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = {}
    for name in ("test", "test_noisy"):
        X, y = featurise(splits[name])
        out[name] = float((model(X).argmax(-1) == y).float().mean())
    return out
