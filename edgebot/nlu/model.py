"""JointBERT-mini: a compact BERT-style Transformer encoder for joint intent
classification and slot filling (Chen et al., 2019, "BERT for Joint Intent
Classification and Slot Filling").

Input tokens are represented by a word embedding plus a character-CNN embedding,
so misspellings and ASR confusions that never appeared in training still get a
meaningful vector. A learned [CLS] token feeds the intent head; every word position
feeds the BIO slot head. The model is small enough (~0.5 M parameters) for
real-time inference on an embedded CPU and exports cleanly to ONNX.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn as nn

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789'"


class Vocab:
    def __init__(self, words: list[str], min_freq: int = 1):
        cnt = Counter(words)
        self.itos = [PAD, UNK, CLS] + sorted(w for w, c in cnt.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, 1) for t in tokens]


def char_ids(tokens: list[str], max_len: int = 16) -> list[list[int]]:
    out = []
    for t in tokens:
        ids = [CHARS.find(ch) + 1 for ch in t[:max_len]]
        out.append(ids + [0] * (max_len - len(ids)))
    return out


class CharCNN(nn.Module):
    def __init__(self, dim: int = 48, out: int = 64):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS) + 1, dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(dim, out // 2, k, padding=k // 2) for k in (3, 5)])

    def forward(self, chars):  # B x T x L
        B, T, L = chars.shape
        x = self.emb(chars.view(B * T, L)).transpose(1, 2)
        h = torch.cat([torch.relu(c(x)).amax(dim=-1) for c in self.convs], dim=-1)
        return h.view(B, T, -1)


class JointBERTMini(nn.Module):
    def __init__(self, vocab_size: int, n_intents: int, n_tags: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ff: int = 256, dropout: float = 0.1, max_len: int = 48):
        super().__init__()
        self.d_model = d_model
        self.word = nn.Embedding(vocab_size, d_model - 64, padding_idx=0)
        self.chars = CharCNN(out=64)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, n_intents))
        self.slot_head = nn.Linear(2 * d_model, n_tags)

    def forward(self, words, chars, mask):
        """words: B x T, chars: B x T x L, mask: B x T (1 = real token).

        Returns intent logits (B x I) and slot logits (B x T x S)."""
        B, T = words.shape
        tok = torch.cat([self.word(words), self.chars(chars)], dim=-1)
        x = torch.cat([self.cls.expand(B, 1, -1), tok], dim=1)
        pos = torch.arange(T + 1, device=words.device)[None]
        x = self.drop(self.norm(x * math.sqrt(self.d_model) / 4 + self.pos(pos)))
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=words.device), mask == 0], dim=1)
        h = self.encoder(x, src_key_padding_mask=key_pad)
        cls_h, tok_h = h[:, 0], h[:, 1:]
        intent = self.intent_head(self.drop(cls_h))
        slot_in = torch.cat([tok_h, cls_h[:, None].expand(-1, T, -1)], dim=-1)
        slots = self.slot_head(self.drop(slot_in))
        return intent, slots


class BoWIntentClassifier(nn.Module):
    """Bag-of-Words multinomial logistic regression baseline (intent only)."""

    def __init__(self, vocab_size: int, n_intents: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, n_intents)

    def forward(self, bow):
        return self.linear(bow)
