"""Synthetic but linguistically varied command corpus for joint intent detection and
slot filling (BIO tagging), in the style of ATIS / SNIPS.

Generalisation is measured honestly: templates are split *before* sampling, so the
test set contains sentence structures never seen in training. An ASR-noise
augmentation (homophones, dropped function words, disfluencies) mimics the errors
a speech recogniser makes so the NLU stays robust in the full voice pipeline.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]
OBJECTS = ["box", "block", "package", "parcel", "crate", "cube", "tote"]
LOCATIONS = ["shelf a", "shelf b", "shelf c", "conveyor", "packing station", "charging dock",
             "loading bay", "workbench"]

# canonical value for words an ASR system is likely to confuse
CANONICAL = {
    "read": "red", "blew": "blue", "grin": "green", "self": "shelf",
    "shelve": "shelf", "conveyer": "conveyor", "packaging": "packing", "bae": "bay", "bey": "bay",
    "boxes": "box", "blocks": "block", "packages": "package", "parcels": "parcel", "crates": "crate",
    "cubes": "cube", "totes": "tote", "doc": "dock", "eh": "a", "bee": "b", "be": "b", "see": "c", "sea": "c",
}

INTENTS = ["navigate", "pick", "place", "pick_place", "stack", "stop", "status", "greet", "help"]
SLOT_TYPES = ["color", "object", "source", "destination"]
TAGS = ["O"] + [f"{p}-{s}" for s in SLOT_TYPES for p in ("B", "I")]

PREFIX = ["", "", "", "please ", "jetbot ", "hey jetbot ", "robot ", "could you ", "can you ", "i need you to ",
          "okay ", "now ", "jetbot please "]
SUFFIX = ["", "", "", " please", " now", " right away", " thanks", " for me", " quickly"]

# {c}=color {o}=object {s}=source location {d}=destination location {c1}{c2}{c3}=stack colours
TEMPLATES = {
    "navigate": [
        "drive me to the {d}", "go and wait at the {d}", "please move over to the {d}", "go straight to the {d}",
        "can you go to the {d}", "move yourself to the {d}", "relocate to the {d}", "reposition at the {d}",
        "steer to the {d}", "come to the {d}", "go park at the {d}", "drive yourself over to the {d}",
        "set course for the {d}", "your destination is the {d}", "go wait by the {d}", "drive up to the {d}",
        "go to the {d}", "drive to the {d}", "move to the {d}", "head to the {d}", "navigate to the {d}",
        "take me to the {d}", "go over to the {d}", "proceed to the {d}", "travel to the {d}",
        "make your way to the {d}", "get to the {d}", "return to the {d}", "go back to the {d}",
        "roll over to the {d}", "visit the {d}", "find the {d}", "drive over to {d}", "go {d}",
        "i want you at the {d}", "your next stop is the {d}", "park at the {d}", "head over to {d}",
    ],
    "pick": [
        "pick up that {c} {o}", "grab hold of the {c} {o}", "snatch the {c} {o}", "pick out the {c} {o}",
        "take hold of the {c} {o} on the {s}", "get the {c} {o} off the {s}", "pull the {c} {o} from the {s}",
        "lift the {c} {o} off the {s}", "seize the {c} {o}", "take up the {c} {o}", "go get the {c} {o}",
        "collect the {c} {o} from the {s}", "please grab the {c} {o} at the {s}",
        "pick up the {c} {o}", "grab the {c} {o}", "pick the {c} {o}", "take the {c} {o}",
        "lift the {c} {o}", "get the {c} {o}", "collect the {c} {o}", "fetch the {c} {o}",
        "grasp the {c} {o}", "hold the {c} {o}", "pick up the {c} {o} from the {s}",
        "grab the {c} {o} on the {s}", "take the {c} {o} off the {s}", "retrieve the {c} {o} from {s}",
        "can you pick the {c} {o} up", "i need the {c} {o}", "get me the {c} {o} from the {s}",
        "pick up the {o} that is {c}", "grab that {c} {o}", "lift up the {c} {o} at the {s}",
    ],
    "place": [
        "put it down on the {d}", "drop that off at the {d}", "drop it off on the {d}", "set it on the {d}",
        "place it down at the {d}", "leave the {c} {o} on the {d}", "unload the {o} onto the {d}",
        "put the {o} down on the {d}", "lower it onto the {d}", "rest it on the {d}", "store it on the {d}",
        "stow it at the {d}", "position it on the {d}",
        "put it on the {d}", "place it on the {d}", "drop it at the {d}", "leave it at the {d}",
        "put the {c} {o} on the {d}", "set it down on the {d}", "place the {o} at the {d}",
        "drop the {c} {o} on the {d}", "release it at the {d}", "put it down at the {d}",
        "deposit it at the {d}", "unload it at the {d}", "place the {c} {o} onto the {d}",
        "set the {c} {o} down at the {d}", "put that on the {d}", "lay it on the {d}",
    ],
    "pick_place": [
        "move the {c} {o} over to the {d}", "carry the {c} {o} from the {s} over to the {d}",
        "take the {c} {o} off the {s} and put it on the {d}", "grab the {c} {o} from the {s} and drop it on the {d}",
        "collect the {c} {o} and deliver it to the {d}", "lift the {c} {o} and place it on the {d}",
        "ship the {c} {o} to the {d}", "send the {c} {o} to the {d}", "get the {c} {o} over to the {d}",
        "pick up the {c} {o} at the {s} and leave it at the {d}", "bring over the {c} {o} to the {d}",
        "transport the {c} {o} to the {d}", "move the {c} {o} from {s} onto the {d}",
        "move the {c} {o} from the {s} to the {d}", "take the {c} {o} from the {s} to the {d}",
        "bring the {c} {o} from the {s} to the {d}", "carry the {c} {o} to the {d}",
        "transfer the {c} {o} from {s} to {d}", "move the {c} {o} to the {d}",
        "bring the {c} {o} to the {d}", "deliver the {c} {o} to the {d}", "take the {c} {o} to the {d}",
        "pick up the {c} {o} and put it on the {d}", "grab the {c} {o} and drop it at the {d}",
        "relocate the {c} {o} from the {s} to the {d}", "shift the {c} {o} over to the {d}",
        "get the {c} {o} from the {s} and bring it to the {d}", "fetch the {c} {o} and leave it at the {d}",
        "the {c} {o} on the {s} needs to go to the {d}", "haul the {c} {o} to the {d}",
        "pick the {c} {o} from the {s} and place it at the {d}",
    ],
    "stack": [
        "stack {c1} {c2} and {c3}", "build a tower of {c1} {c2} {c3}", "stack the {c1} the {c2} and the {c3} blocks",
        "put {c1} at the bottom then {c2} then {c3}", "build the tower {c1} {c2} {c3}",
        "make a tower with the {c1} {c2} and {c3} cubes", "pile up {c1} {c2} and {c3}",
        "stack {c1} then {c2} then {c3} on the {d}", "assemble a stack of {c1} {c2} {c3}",
        "construct a tower from {c1} {c2} and {c3}", "place {c1} {c2} and {c3} in a stack",
        "stack the {c1} {c2} and {c3} blocks", "build a tower with {c1} {c2} and {c3}",
        "stack {c1} on {c2} on {c3}", "assemble the {c1} {c2} and {c3} blocks into a tower",
        "make a stack of {c1} {c2} {c3}", "stack the blocks {c1} then {c2} then {c3}",
        "put {c1} {c2} and {c3} on top of each other", "build a stack {c1} {c2} {c3} on the {d}",
        "assemble {c1} {c2} and {c3} at the {d}", "create a tower of {c1} {c2} and {c3} blocks",
        "pile the {c1} {c2} and {c3} cubes", "stack {c1} {c2} {c3} at the {d}",
        "assemble a tower using the {c1} {c2} and {c3} blocks", "stack up {c1} and {c2} and {c3}",
    ],
    "stop": [
        "stop that", "stop immediately", "halt now", "cancel", "cancel everything", "abort mission",
        "stop the robot", "do not move", "please stop", "wait", "hold it", "stop stop stop", "terminate the task",
        "forget it", "cease", "stop what you are doing", "kill the task",
        "stop", "stop now", "halt", "freeze", "stop moving", "emergency stop", "cancel that", "abort",
        "abort the task", "hold on", "wait stop", "stop right there", "cancel the mission", "never mind",
        "stand still", "pause", "quit what you are doing", "that is enough stop",
    ],
    "status": [
        "what is going on", "are you busy", "how much battery do you have", "where are you now",
        "what is your current location", "are you finished", "what is the progress", "report",
        "give me a status update", "what is happening", "update me", "what is your position",
        "what did you just do", "is everything okay", "are you carrying something",
        "what are you doing", "status report", "what is your status", "where are you", "how is it going",
        "report your status", "what is your battery level", "are you done", "how far along are you",
        "give me an update", "what is the current task", "what are you carrying", "are you holding anything",
        "tell me your position", "status", "show me your progress", "is the task complete",
    ],
    "greet": [
        "hey jetbot", "hello robot", "hi robot", "hello there", "morning", "good day", "hey buddy",
        "hello hello", "hi jetbot", "what's up", "sup", "hey how are you", "hello friend", "evening",
        "hello", "hi", "hey", "good morning", "hello jetbot", "hi there", "hey robot", "good afternoon",
        "good evening", "greetings", "hey there", "howdy", "nice to meet you", "hiya", "yo jetbot",
    ],
    "help": [
        "help me", "please help", "show me the commands", "what do you understand", "tell me what you can do",
        "what are the commands", "how can you help", "what tasks can you do", "instructions please",
        "what should i say", "i do not know what to say", "help me out", "what are my options", "teach me",
        "help", "what can you do", "how do i use you", "what commands do you know", "list your commands",
        "i need help", "show me what you can do", "what are your abilities", "how does this work",
        "give me some examples", "what can i ask you", "can you help me", "explain your features",
    ],
}

HOMOPHONES = {"red": ["read"], "blue": ["blew"], "shelf": ["self", "shelve"], "to": ["too", "two"],
              "conveyor": ["conveyer"], "bay": ["bey"], "green": ["grin"], "a": ["eh"], "b": ["bee", "be"],
              "c": ["see", "sea"], "dock": ["doc"]}
FUNCTION_WORDS = {"the", "a", "to", "it", "on", "at"}
FILLERS = ["uh", "um", "er", "like"]


@dataclass
class Example:
    tokens: list
    tags: list
    intent: str

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def split_templates(seed: int = 7, test_frac: float = 0.25):
    rng = random.Random(seed)
    train, test = {}, {}
    for intent, temps in TEMPLATES.items():
        t = temps[:]
        rng.shuffle(t)
        k = max(2, int(round(len(t) * test_frac)))
        test[intent], train[intent] = t[:k], t[k:]
    return train, test


def _fill(template: str, rng: random.Random):
    """Expands a template into (tokens, tags)."""
    colors = rng.sample(COLORS, 3)
    src, dst = rng.sample(LOCATIONS, 2)
    values = {"c": (rng.choice(COLORS), "color"), "o": (rng.choice(OBJECTS), "object"),
              "s": (src, "source"), "d": (dst, "destination"),
              "c1": (colors[0], "color"), "c2": (colors[1], "color"), "c3": (colors[2], "color")}
    tokens, tags = [], []
    for piece in re.split(r"(\{\w+\})", template):
        if not piece:
            continue
        m = re.fullmatch(r"\{(\w+)\}", piece)
        if m:
            val, slot = values[m.group(1)]
            for i, w in enumerate(val.split()):
                tokens.append(w)
                tags.append(f"{'B' if i == 0 else 'I'}-{slot}")
        else:
            for w in piece.split():
                tokens.append(w)
                tags.append("O")
    return tokens, tags


def asr_noise(tokens, tags, rng: random.Random, p: float = 0.12):
    out_t, out_g = [], []
    for w, g in zip(tokens, tags):
        r = rng.random()
        if g == "O" and w in FUNCTION_WORDS and r < p:
            continue  # dropped function word
        if w in HOMOPHONES and r < p * 2:
            w = rng.choice(HOMOPHONES[w])
        if rng.random() < p / 4:
            out_t.append(rng.choice(FILLERS))
            out_g.append("O")
        out_t.append(w)
        out_g.append(g)
    return out_t, out_g


def generate(templates: dict, n_per_intent: int, seed: int, noise: float = 0.0) -> list:
    rng = random.Random(seed)
    data = []
    for intent, temps in templates.items():
        for _ in range(n_per_intent):
            toks, tags = _fill(rng.choice(temps), rng)
            pre = rng.choice(PREFIX).split()
            suf = rng.choice(SUFFIX).split()
            toks = pre + toks + suf
            tags = ["O"] * len(pre) + tags + ["O"] * len(suf)
            if noise > 0 and rng.random() < 0.5:
                toks, tags = asr_noise(toks, tags, rng, noise)
            data.append(Example(toks, tags, intent))
    rng.shuffle(data)
    return data


def build_splits(seed: int = 7, n_train: int = 1200, n_test: int = 300):
    train_t, test_t = split_templates(seed)
    train = generate(train_t, n_train, seed, noise=0.12)
    val = generate(train_t, n_test // 2, seed + 1, noise=0.12)
    test = generate(test_t, n_test, seed + 2, noise=0.0)           # unseen templates, clean
    test_noisy = generate(test_t, n_test, seed + 3, noise=0.2)     # unseen templates + ASR noise
    return {"train": train, "val": val, "test": test, "test_noisy": test_noisy}


def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def canonical(word: str) -> str:
    return CANONICAL.get(word, word)
