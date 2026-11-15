"""Dialogue-manager behaviour with a scripted NLU and a scripted controller."""

import numpy as np

from edgebot.nlu.engine import Frame
from edgebot.robot.agent import EdgeBot


class ScriptedNLU:
    def __init__(self, frames):
        self.frames = list(frames)

    def parse(self, text):
        return self.frames.pop(0)


def homing_policy(bot):
    def policy(obs):
        sin_b, cos_b = obs[25], obs[26]
        return np.array([0.2 if cos_b > 0.8 else -1.0, float(np.clip(2.0 * np.arctan2(sin_b, cos_b), -1, 1))])
    return policy


def make_bot(frames):
    bot = EdgeBot(ScriptedNLU(frames), None)
    bot.policy = homing_policy(bot)
    return bot


def test_clarification_and_context():
    bot = make_bot([Frame("pick", 0.9, {"object": "box"}), Frame("pick", 0.9, {"color": "blue"}),
                    Frame("status", 0.9, {})])
    assert "colour" in bot.handle("pick up the box").response
    turn = bot.handle("the blue one")
    assert turn.success and bot.holding == ("blue", "crate")
    assert "carrying the blue crate" in bot.handle("status").response


def test_low_confidence_is_rejected():
    bot = make_bot([Frame("navigate", 0.2, {"destination": "conveyor"})])
    assert "rephrase" in bot.handle("mumble").response
    assert bot.station == "charging dock"


def test_missing_item_reported():
    bot = make_bot([Frame("pick", 0.9, {"color": "purple", "source": "shelf a"})])
    turn = bot.handle("pick the purple tote from shelf a")
    assert turn.success is False and "could not find" in turn.response
