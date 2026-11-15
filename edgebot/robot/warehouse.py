"""The warehouse used for full missions: fixed fixtures, named stations and an
initial inventory of coloured items on each station's work surface."""

from __future__ import annotations

import copy

import numpy as np

from edgebot.envs.jetbot_nav import Scenario

WIDTH, HEIGHT = 6.0, 5.0

FIXTURES = {
    "shelf a": [0.6, 4.25, 2.4, 4.65],
    "shelf b": [3.4, 4.25, 5.2, 4.65],
    "shelf c": [1.9, 2.15, 3.1, 2.5],
    "rack": [3.9, 1.6, 4.3, 3.0],
    "conveyor": [5.55, 0.9, 5.95, 3.4],
    "packing station": [2.4, 0.05, 3.6, 0.4],
    "workbench": [4.4, 0.05, 5.3, 0.35],
    "charging dock": [0.05, 0.05, 0.35, 0.35],
}

# docking point (x, y) in front of each fixture
STATIONS = {
    "shelf a": (1.5, 3.8),
    "shelf b": (4.3, 3.8),
    "shelf c": (2.5, 2.95),
    "conveyor": (5.1, 2.1),
    "packing station": (3.0, 0.85),
    "workbench": (4.85, 0.8),
    "charging dock": (0.65, 0.65),
    "loading bay": (0.6, 2.4),
}

OBSTACLES = np.array([
    [1.0, 1.3, 0.22],   # pallet
    [3.2, 3.4, 0.16],   # stray box
    [0.9, 3.1, 0.15],   # traffic cone
])

INITIAL_INVENTORY = {
    "shelf a": [("red", "box"), ("green", "block")],
    "shelf b": [("blue", "crate"), ("yellow", "parcel")],
    "shelf c": [("orange", "cube"), ("purple", "tote")],
    "workbench": [("red", "block"), ("green", "block"), ("blue", "block")],
    "conveyor": [],
    "packing station": [],
    "loading bay": [("yellow", "box")],
    "charging dock": [],
}

TABLE_SLOTS = (0.30, 0.40, 0.50, 0.24, 0.56)


def scenario() -> Scenario:
    boxes = np.array([v for k, v in FIXTURES.items()])
    return Scenario(WIDTH, HEIGHT, boxes, OBSTACLES.copy(), start=np.array([*STATIONS["charging dock"], 0.8]),
                    goal=np.array(STATIONS["shelf a"]))


def initial_inventory() -> dict:
    return copy.deepcopy(INITIAL_INVENTORY)
