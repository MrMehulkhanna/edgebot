"""Knowledge distillation from the fine-tuned JointBERT teacher into JointBERT-mini.

Following TinyBERT (Jiao et al., 2020), the transfer set is expanded with
masked-language-model augmentation: non-slot words of training sentences are
masked and replaced by DistilBERT's top MLM predictions ("haul" -> "move", "carry",
"bring" ...). The teacher labels these new sentences with soft intent and slot
distributions and the student minimises the temperature-scaled KL divergence to
them, inheriting the teacher's lexical knowledge at ~1/100 of its size.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgebot.nlu.data import INTENTS, TAGS, Example
from edgebot.nlu.engine import NLU, _targets, encode_batch
from edgebot.nlu.model import JointBERTMini, Vocab


@torch.no_grad()
def mlm_augment(examples: list[Example], n_variants: int = 3, mlm_name: str = "distilbert-base-uncased",
                device="cuda", topk: int = 8, max_masks: int = 2, seed: int = 0, bs: int = 64) -> list[Example]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(mlm_name)
    mlm = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device).eval()
    stop = set("the a an to it on at of and from".split())
    jobs = []
    for e in examples:
        cand = [i for i, (w, t) in enumerate(zip(e.tokens, e.tags)) if t == "O" and w not in stop]
        if not cand:
            continue
        for _ in range(n_variants):
            k = int(rng.integers(1, min(max_masks, len(cand)) + 1))
            jobs.append((e, sorted(rng.choice(cand, size=k, replace=False).tolist())))
    out = []
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        masked = [[tok.mask_token if j in pos else w for j, w in enumerate(e.tokens)] for e, pos in chunk]
        enc = tok(masked, is_split_into_words=True, return_tensors="pt", padding=True).to(device)
        logits = mlm(**enc).logits
        for b, (e, pos) in enumerate(chunk):
            new = list(e.tokens)
            mask_idx = (enc["input_ids"][b] == tok.mask_token_id).nonzero().flatten().tolist()
            for p_word, p_tok in zip(pos, mask_idx):
                top = logits[b, p_tok].topk(topk * 3).indices.tolist()
                words = [tok.convert_ids_to_tokens(t) for t in top]
                words = [w for w in words if w.isalpha() and len(w) > 1 and w != e.tokens[p_word]][:topk]
                if words:
                    new[p_word] = words[int(rng.integers(len(words)))]
            if new != e.tokens:
                out.append(Example(new, list(e.tags), e.intent))
    return out


def distill(teacher, splits: dict, augmented: list[Example], device="cuda", epochs: int = 30, T: float = 2.0,
            alpha: float = 0.5, lr: float = 2e-3, bs: int = 64, seed: int = 7, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gold = splits["train"]
    transfer = gold + augmented
    _, _, ip, sp = teacher.predict([e.tokens for e in transfer], device)
    vocab = Vocab([w for e in transfer for w in e.tokens], min_freq=2)
    student = JointBERTMini(len(vocab), len(INTENTS), len(TAGS)).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(transfer) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    is_gold = np.array([1.0] * len(gold) + [0.0] * len(augmented))
    nlu = NLU(student, vocab, device)
    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        student.train()
        perm = np.random.permutation(len(transfer))
        tot = 0.0
        for i in range(0, len(transfer), bs):
            idx = perm[i:i + bs]
            batch = [transfer[j] for j in idx]
            words, chars, mask = encode_batch([e.tokens for e in batch], vocab, device, word_dropout=0.05)
            il, sl = student(words, chars, mask)
            Tlen = words.shape[1]
            t_int = torch.tensor(np.stack([ip[j] for j in idx]), device=device)
            t_slot = torch.zeros(len(batch), Tlen, len(TAGS), device=device)
            for k, j in enumerate(idx):
                t_slot[k, :len(sp[j])] = torch.as_tensor(sp[j], device=device)
            # soft targets (teacher), temperature-scaled
            kd_int = F.kl_div(F.log_softmax(il / T, -1), (t_int.log() / T).softmax(-1), reduction="batchmean") * T * T
            m = mask.float()
            soft_slot = (t_slot.clamp_min(1e-8).log() / T).softmax(-1)
            kd_slot = (F.kl_div(F.log_softmax(sl / T, -1), soft_slot, reduction="none").sum(-1) * m).sum() / m.sum()
            loss = kd_int + kd_slot * T * T
            # hard targets on the human-labelled part of the batch
            g = torch.tensor(is_gold[idx], device=device)
            if g.sum() > 0:
                yi, yt = _targets(batch, device)
                ce_int = (F.cross_entropy(il, yi, reduction="none") * g).sum() / g.sum()
                yt = torch.where(g[:, None].bool(), yt, torch.full_like(yt, -100))
                ce_slot = F.cross_entropy(sl.reshape(-1, len(TAGS)), yt.reshape(-1), ignore_index=-100)
                loss = alpha * loss + (1 - alpha) * (ce_int + ce_slot)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
        student.eval()
        val = nlu.evaluate(splits["val"])
        score = val["intent_acc"] + val["slot_f1"]
        if score >= best:
            best = score
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        log(f"[KD] epoch {ep:2d} | loss {tot / len(transfer):.4f} | val intent {val['intent_acc']:.4f} "
            f"| val slot-F1 {val['slot_f1']:.4f} | {time.time() - t0:.0f}s")
    student.load_state_dict(best_state)
    student.eval()
    return nlu
