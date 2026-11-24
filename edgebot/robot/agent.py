"""EdgeBot: the integrated voice-commanded mobile manipulator.

    microphone / WAV -> ASR (Whisper) -> NLU (JointBERT-mini) -> dialogue manager
        -> task planner -> [A* global path + RL local policy]  (navigation)
                         -> [IK + min-jerk trajectories + PD]  (manipulation)
        -> natural-language feedback -> TTS (Piper) -> speaker / WAV

The dialogue manager keeps context across turns: it asks for missing slots,
resolves "it" to the carried item, checks the inventory before acting and reports
failures honestly.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field

import numpy as np

from edgebot.envs.jetbot_nav import JetBotNavEnv
from edgebot.manipulation.workcell import Workcell
from edgebot.robot import warehouse as wh
from edgebot.robot.planner import plan_path

CONFIDENCE_THRESHOLD = 0.55


def pretty(station: str) -> str:
    """'shelf a' -> 'shelf A' for spoken / printed responses."""
    words = station.split()
    return " ".join(w.upper() if len(w) == 1 else w for w in words)

HELP_TEXT = ("I can drive to any station, pick up and deliver items, and stack blocks. Try: go to shelf B, "
             "move the red box from shelf A to the conveyor, or stack the red, green and blue blocks.")


@dataclass
class Turn:
    user: str
    frame: dict
    response: str
    success: bool | None = None
    timings: dict = field(default_factory=dict)
    asr_text: str | None = None


class FrameSink:
    """Collects composited video frames (map view | arm view | dialogue panel)."""

    def __init__(self, enabled: bool = True, map_every: int = 2, arm_every: int = 6):
        self.enabled = enabled
        self.frames: list = []
        self.map_every, self.arm_every = map_every, arm_every
        self.map_img = None
        self.arm_img = None
        self.lines: list = []

    def caption(self, speaker: str, text: str):
        self.lines.append((speaker, text))
        self.lines = self.lines[-8:]

    def compose(self):
        if not self.enabled:
            return
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default(size=14)
        small = ImageFont.load_default(size=12)
        canvas = Image.new("RGB", (1140, 640), (255, 255, 255))
        if self.map_img is not None:
            canvas.paste(Image.fromarray(self.map_img).resize((600, 500)), (10, 56))
        if self.arm_img is not None:
            canvas.paste(Image.fromarray(self.arm_img).resize((510, 392)), (620, 56))
        d = ImageDraw.Draw(canvas)
        title = "EdgeBot  |  RL navigation + IK manipulation + conversational AI"
        d.text((12, 10), title, fill=(20, 20, 20), font=font)
        d.text((12, 34), "Warehouse map (A* plan, PPO local control, 24-beam lidar)", fill=(90, 90, 90), font=small)
        d.text((622, 34), "Onboard arm workcell (side view)", fill=(90, 90, 90), font=small)
        d.rectangle([620, 456, 1130, 630], outline=(200, 200, 200))
        y = 462
        for speaker, text in self.lines:
            color = (30, 90, 200) if speaker == "User" else (20, 130, 60)
            for line in textwrap.wrap(f"{speaker}: {text}", 68, subsequent_indent="    "):
                d.text((628, y), line, fill=color, font=small)
                y += 16
            if y > 612:
                break
        d.text((12, 566), "Purple line: A* global plan  |  Blue: driven path  |  Red rays: lidar  |  Green ring: goal",
               fill=(90, 90, 90), font=small)
        self.frames.append(np.asarray(canvas))


class EdgeBot:
    def __init__(self, nlu, policy, asr=None, tts=None, sink: FrameSink | None = None, seed: int = 0):
        self.nlu, self.policy, self.asr, self.tts = nlu, policy, asr, tts
        self.use_shield = True
        self.sink = sink or FrameSink(enabled=False)
        self.scn = wh.scenario()
        self.env = JetBotNavEnv(fixed_scenario=self.scn, max_steps=100_000)
        self.env.reset(seed=seed)
        self.inventory = wh.initial_inventory()
        self.pose = np.array([*wh.STATIONS["charging dock"], 0.8])
        self.station = "charging dock"
        self.holding: tuple | None = None
        self.pending: dict | None = None
        self.last_task = "waiting for instructions"
        self.distance_travelled = 0.0
        self.battery = 100.0
        self.history: list[Turn] = []
        self.stats = {"nav_attempts": 0, "nav_success": 0, "collisions": 0, "picks": 0, "places": 0,
                      "manip_failures": 0, "nav_steps": 0}
        self._refresh_views()

    # ================================================================ I/O
    def hear(self, audio) -> Turn:
        assert self.asr is not None, "ASR model not loaded"
        text, t_asr = self.asr.transcribe(audio)
        turn = self.handle(text)
        turn.asr_text = text
        turn.timings["asr_s"] = t_asr
        return turn

    def speak(self, text: str, path=None) -> float | None:
        if self.tts is None:
            return None
        t0 = time.perf_counter()
        if path is not None:
            self.tts.save(text, path)
        else:
            self.tts.synthesize(text)
        return time.perf_counter() - t0

    def handle(self, text: str) -> Turn:
        self.sink.caption("User", text)
        t0 = time.perf_counter()
        frame = self.nlu.parse(text)
        t_nlu = time.perf_counter() - t0
        t1 = time.perf_counter()
        response, success = self._dispatch(frame)
        t_exec = time.perf_counter() - t1
        self.sink.caption("EdgeBot", response)
        for _ in range(12):
            self.sink.compose()
        turn = Turn(text, frame.to_dict(), response, success, {"nlu_ms": t_nlu * 1000, "execute_s": t_exec})
        self.history.append(turn)
        return turn

    # ================================================================ dialogue
    def _merge_pending(self, frame):
        """Fills a previously incomplete request with slots from a follow-up answer."""
        if self.pending is None:
            return frame.intent, dict(frame.slots)
        if frame.intent in ("stop", "greet", "help", "status") and frame.confidence > CONFIDENCE_THRESHOLD:
            self.pending = None
            return frame.intent, dict(frame.slots)
        intent = self.pending["intent"]
        slots = {**self.pending["slots"], **{k: v for k, v in frame.slots.items() if k not in self.pending["slots"]}}
        if "destination" in frame.slots and intent in ("pick_place", "place", "navigate", "stack"):
            slots["destination"] = frame.slots["destination"]
        self.pending = None
        return intent, slots

    def _dispatch(self, frame):
        if frame.confidence < CONFIDENCE_THRESHOLD and self.pending is None:
            return "Sorry, I did not quite catch that. Could you rephrase the command?", None
        intent, slots = self._merge_pending(frame)

        if intent == "greet":
            return "Hello! I am EdgeBot, your warehouse assistant. How can I help?", True
        if intent == "help":
            return HELP_TEXT, True
        if intent == "status":
            return self._status(), True
        if intent == "stop":
            self.pending = None
            return "Stopping. All motion halted and the current task is cancelled.", True
        if intent == "navigate":
            dest = slots.get("destination")
            if dest not in wh.STATIONS:
                return self._ask(intent, slots, "Where should I go? For example shelf A or the conveyor.")
            ok, msg = self.navigate(dest)
            return msg, ok
        if intent == "pick":
            return self._pick(slots)
        if intent == "place":
            return self._place(slots)
        if intent == "pick_place":
            if "color" not in slots:
                return self._ask(intent, slots, "Which item should I move? Tell me its colour.")
            if slots.get("destination") not in wh.STATIONS:
                return self._ask(intent, slots, f"Where should I take the {self._desc(slots)}?")
            msg, ok = self._pick(slots)
            if not ok:
                return msg, ok
            msg2, ok2 = self._place(slots)
            return f"{msg} {msg2}", ok2
        if intent == "stack":
            return self._stack(slots)
        return "I am not sure how to do that yet. Say help to hear what I can do.", None

    def _ask(self, intent, slots, question):
        self.pending = {"intent": intent, "slots": slots}
        return question, None

    @staticmethod
    def _desc(slots) -> str:
        return " ".join(x for x in (slots.get("color"), slots.get("object", "item")) if x)

    def _status(self) -> str:
        held = f"carrying the {' '.join(self.holding)}" if self.holding else "not carrying anything"
        return (f"I am at the {pretty(self.station)}, {held}. Battery at {self.battery:.0f} percent, "
                f"{self.distance_travelled:.1f} metres travelled. Last task: {self.last_task}.")

    # ================================================================ skills
    def _find_item(self, color, obj=None, source=None):
        stations = [source] if source in self.inventory else list(self.inventory)
        for match_type in (True, False):
            for st in stations:
                for item in self.inventory[st]:
                    if item[0] == color and (not match_type or obj is None or item[1] == obj):
                        return st, item
        return None, None

    def _pick(self, slots):
        if self.holding:
            return f"I am already carrying the {' '.join(self.holding)}. Tell me where to put it first.", False
        color = slots.get("color")
        if color is None:
            return self._ask("pick", slots, "Which colour item should I pick up?")
        st, item = self._find_item(color, slots.get("object"), slots.get("source"))
        if item is None:
            where = f" on the {pretty(slots['source'])}" if slots.get("source") else ""
            return f"I could not find a {self._desc(slots)}{where} in the inventory.", False
        ok, msg = self.navigate(st)
        if not ok:
            return msg, False
        cell = self._workcell(st)
        name = f"{item[0]}_{item[1]}"
        self.last_task = f"picking the {item[0]} {item[1]} at the {pretty(st)}"
        if not cell.pick(name):
            self.stats["manip_failures"] += 1
            return f"The grasp on the {item[0]} {item[1]} failed. Please check the item.", False
        cell.move_joint(cell.q_home, 1.2, label="stow")
        self.inventory[st].remove(item)
        self.holding = item
        self.stats["picks"] += 1
        return f"Picked up the {item[0]} {item[1]} from the {pretty(st)}.", True

    def _place(self, slots):
        if not self.holding:
            return "I am not holding anything to place.", False
        dest = slots.get("destination")
        if dest not in wh.STATIONS:
            return self._ask("place", slots, f"Where should I put the {' '.join(self.holding)}?")
        if len(self.inventory[dest]) >= len(wh.TABLE_SLOTS):
            return f"The {pretty(dest)} is full.", False
        ok, msg = self.navigate(dest)
        if not ok:
            return msg, False
        cell = self._workcell(dest, carrying=self.holding)
        used = {round(b.x, 2) for b in cell.blocks.values() if not b.held}
        x = next(s for s in wh.TABLE_SLOTS if round(s, 2) not in used)
        item = self.holding
        self.last_task = f"delivering the {item[0]} {item[1]} to the {pretty(dest)}"
        if not cell.place(x):
            self.stats["manip_failures"] += 1
            return f"I could not place the {item[0]} {item[1]} at the {pretty(dest)}.", False
        cell.home()
        self.inventory[dest].append(item)
        self.holding = None
        self.stats["places"] += 1
        return f"Delivered the {item[0]} {item[1]} to the {pretty(dest)}.", True

    def _stack(self, slots):
        colors = slots.get("colors") or ([slots["color"]] if "color" in slots else [])
        if len(colors) < 2:
            return self._ask("stack", slots, "Which blocks should I stack? Name the colours from bottom to top.")
        if self.holding:
            return f"Please let me put down the {' '.join(self.holding)} first.", False
        station = slots.get("destination") if slots.get("destination") in self.inventory else None
        candidates = [station] if station else list(self.inventory)
        st = next((s for s in candidates if all(any(i[0] == c for i in self.inventory[s]) for c in colors)), None)
        if st is None:
            return f"I cannot find {', '.join(colors)} blocks together at one station.", False
        ok, msg = self.navigate(st)
        if not ok:
            return msg, False
        cell = self._workcell(st)
        names = []
        for c in colors:
            item = next(i for i in self.inventory[st] if i[0] == c)
            names.append(f"{item[0]}_{item[1]}")
        free = [s for s in wh.TABLE_SLOTS if all(abs(b.x - s) > 0.05 for b in cell.blocks.values())]
        if not free:
            return f"There is no free space on the {pretty(st)} to build the tower.", False
        x = free[0]
        self.last_task = f"stacking {', '.join(colors)} at the {pretty(st)}"
        ok = cell.stack(names, x) and cell.tower_ok(names, x)
        if not ok:
            self.stats["manip_failures"] += 1
            return "The tower assembly failed during the stacking sequence.", False
        return f"Built a tower of {', '.join(colors)} blocks at the {pretty(st)}, bottom to top.", True

    # ================================================================ navigation
    def navigate(self, dest: str, max_steps: int = 3000):
        if dest == self.station:
            return True, f"I am already at the {pretty(dest)}."
        goal = np.array(wh.STATIONS[dest])
        self.stats["nav_attempts"] += 1
        wps = plan_path(self.scn, self.pose[:2], goal, inflate=0.24, max_seg=1.0)
        if wps is None:
            return False, f"I could not find a path to the {pretty(dest)}."
        env = self.env
        from edgebot.rl.shield import LidarShield

        shield = LidarShield(env)  # novelty: lidar safety filter between policy and motors
        obs, _ = env.reset(options={"scenario": self.scn, "start": self.pose, "goal": wps[0]})
        self._plan = [self.pose[:2].copy()] + wps
        k = 0
        start = self.pose.copy()
        self.last_task = f"driving to the {pretty(dest)}"
        for step in range(max_steps):
            obs, _, term, _, info = env.step(shield(self.policy(obs)) if self.use_shield else self.policy(obs))
            if k < len(wps) - 1 and info["distance"] < 0.35:
                k += 1
                env.set_goal(wps[k])
                obs = env._obs()
            if step % self.sink.map_every == 0:
                self._refresh_views(nav=True)
                self.sink.compose()
            if info["collision"]:
                self.stats["collisions"] += 1
                self.pose = env.pose.copy()
                self._account(env, start)
                return False, f"I bumped into an obstacle on the way to the {pretty(dest)} and stopped for safety."
            if k == len(wps) - 1 and info["success"]:
                self.pose = env.pose.copy()
                self.station = dest
                self.stats["nav_success"] += 1
                self._account(env, start)
                self._refresh_views(nav=True)
                return True, f"Arrived at the {pretty(dest)}."
        self.pose = env.pose.copy()
        self._account(env, start)
        return False, f"I could not reach the {pretty(dest)} in time."

    def _account(self, env, start):
        trail = np.array(env.trail)
        dist = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1))) if len(trail) > 1 else 0.0
        self.distance_travelled += dist
        self.battery = max(0.0, self.battery - 1.5 * dist)
        self.stats["nav_steps"] += len(trail) - 1

    # ================================================================ manipulation
    def _workcell(self, station: str, carrying=None) -> Workcell:
        cell = Workcell()
        for item, x in zip(self.inventory[station], wh.TABLE_SLOTS):
            cell.add_block(f"{item[0]}_{item[1]}", item[0], x)
        if carrying is not None:
            name = f"{carrying[0]}_{carrying[1]}"
            b = cell.add_block(name, carrying[0], 0.0)
            b.held = True
            cell.holding = name
            pose = cell.ee()
            b.x, b.z = pose[0], pose[1] - b.h * 0.6
        k = [0]

        def on_step(label):
            k[0] += 1
            if k[0] % self.sink.arm_every == 0:
                self.sink.arm_img = cell.render(title=f"{station}: {label}")
                self.sink.compose()

        cell.on_step = on_step if self.sink.enabled else None
        if self.sink.enabled:
            self.sink.arm_img = cell.render(title=station)
        return cell

    # ================================================================ views
    def _refresh_views(self, nav: bool = False):
        if not self.sink.enabled:
            return
        from PIL import Image, ImageDraw

        env = self.env
        if not nav:
            env.reset(options={"scenario": self.scn, "start": self.pose, "goal": self.pose[:2]})
        img = Image.fromarray(env.render(scale=100, show_lidar=nav))
        d = ImageDraw.Draw(img)
        H = img.size[1]
        for name, (x, y) in wh.STATIONS.items():
            d.ellipse([x * 100 - 4, H - y * 100 - 4, x * 100 + 4, H - y * 100 + 4], fill=(120, 60, 160))
            d.text((x * 100 + 6, H - y * 100 - 6), name, fill=(70, 30, 110))
        if nav and getattr(self, "_plan", None) is not None:
            d.line([(p[0] * 100, H - p[1] * 100) for p in self._plan], fill=(160, 60, 200), width=1)
        if self.holding:
            d.text((8, 8), f"carrying: {' '.join(self.holding)}", fill=(200, 40, 40))
        self.sink.map_img = np.asarray(img)
