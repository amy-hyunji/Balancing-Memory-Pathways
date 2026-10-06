import re
import unicodedata

import gymnasium as gym
import minigrid  # noqa: F401  (registers BabyAI-*-v0)
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

SEED_BASE = {"train": 0, "valid": 20000, "test": 30000}

ACTIONS = [
    ("turn left",  0, ("left", "turn left")),
    ("turn right", 1, ("right", "turn right")),
    ("go forward", 2, ("forward", "go forward", "move forward", "step forward")),
    ("pick up",    3, ("pickup", "pick up", "pick up object", "take")),
    ("drop",       4, ("drop", "put down", "drop object")),
    ("toggle",     5, ("toggle", "open", "open door", "unlock", "close")),
]
NAMES = [a[0] for a in ACTIONS]
NOOP = 6  # minigrid's `done`: accepted, changes nothing

GOAL_PREFIX = ("You are an expert BabyAI agent moving through a gridworld. Your goal is to "
               "generate the best next action that will lead to completing the mission.\n\n"
               "End your output sequence with an action.\n\nHere is your mission:\n{task}\n\n"
               "Here is your interactions so far:")
SUFFIX = "\n\nOutput the final action directly within <action> </action> tags."
RULES = ("You see a 7x7 patch of the grid in front of you; you cannot see through "
         "walls or closed doors. 'go forward' moves one cell ahead (blocked by walls, "
         "closed doors and objects), 'turn left'/'turn right' rotate you in place, "
         "'pick up' takes the object in the cell directly ahead, 'drop' puts a carried "
         "object into that cell, and 'toggle' opens/closes a door there (a locked door "
         "needs the matching key in hand).\n")


def game_specs(level, split, n):
    """level: e.g. MiniBossLevel, PutNextLocal, GoToObjMaze."""
    base = SEED_BASE[split]
    return [{"env": "babyai", "game": level, "id": f"{level}_{base + i}", "seed": base + i}
            for i in range(1, n + 1)]


def _cell(fwd, right):
    parts = []
    if fwd:
        parts.append(f"{fwd} step{'s' if fwd != 1 else ''} forward")
    if right:
        n = abs(right)
        parts.append(f"{n} step{'s' if n != 1 else ''} {'right' if right > 0 else 'left'}")
    return " and ".join(parts) if parts else "where you stand"


def _obj(t, c, s):
    name, color = IDX_TO_OBJECT[int(t)], IDX_TO_COLOR[int(c)]
    if name == "door":
        return f"a {({0: 'open', 1: 'closed', 2: 'locked'}.get(int(s), ''))} {color} door".replace("  ", " ")
    return f"a {color} {name}"


def describe(obs, env):
    img = obs["image"]                      # img[i][j]: i left->right, j far->near, agent at (3, 6)
    objs, walls_fwd, walls_side = [], [], []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            t, c, s = img[i, j]
            name = IDX_TO_OBJECT[int(t)]
            fwd, right = 6 - j, i - 3
            if (i, j) == (3, 6) or name in ("unseen", "empty"):
                continue
            if name == "wall":
                if right == 0 and fwd > 0:
                    walls_fwd.append(fwd)
                elif fwd == 0 and right != 0:
                    walls_side.append(right)
                continue
            objs.append((fwd, abs(right), _obj(t, c, s), right))
    lines = ["You see:"] + [f"- {p} {_cell(f, r)}" for f, _, p, r in sorted(objs, key=lambda o: (o[0], o[1]))] \
        if objs else ["You see: nothing of interest."]
    if walls_fwd:
        lines.append(f"- a wall {min(walls_fwd)} steps forward")
    for r in sorted(walls_side, key=abs)[:2]:
        lines.append(f"- a wall {abs(r)} step{'s' if abs(r) != 1 else ''} {'right' if r > 0 else 'left'}")
    t, c, s = img[3, 5]                     # the cell 'go forward' would enter
    ahead = IDX_TO_OBJECT[int(t)]
    front = ("nothing (you can move there)" if ahead in ("empty", "unseen")
             else "a wall" if ahead == "wall" else _obj(t, c, s))
    lines.append(f"Directly in front of you: {front}")
    carrying = getattr(env.unwrapped, "carrying", None)
    lines.append("You are carrying: " + (f"a {carrying.color} {carrying.type}" if carrying else "nothing"))
    lines.append("Admissible commands: " + ", ".join(NAMES))
    return "\n".join(lines)


def _normalize(s):
    s = unicodedata.normalize("NFKC", (s or "").lower().strip())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", s)).strip()


def snap(action):
    """-> (action id, status); unrecognized actions become a no-op step."""
    if action is None:
        return NOOP, "no_action_tag"
    n = _normalize(action)
    for name, idx, _ in ACTIONS:
        if n == name:
            return idx, "valid"
    for name, idx, aliases in ACTIONS:
        if n in aliases or any(n.startswith(a + " ") for a in aliases):
            return idx, "alias"
    return NOOP, "invalid"


class BabyAIEnv:
    parse_last = False          # the first <action> is the one to take (models sometimes write a plan)
    max_new_tokens = 512

    def __init__(self, spec):
        self.env = gym.make(f"BabyAI-{spec['game']}-v0")
        self.seed = spec["seed"]
        self.bot, self.prev = None, None

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.max_steps = self.env.unwrapped.max_steps
        return (GOAL_PREFIX.format(task=obs["mission"]) + "\n" + RULES
                + f"\ncurrent state: {describe(obs, self.env)}" + SUFFIX)

    def step(self, action):
        """-> (next user message, won, done, status)"""
        aid, status = snap(action)
        self.prev = aid
        obs, reward, term, trunc, _ = self.env.step(aid)
        return f"current state: {describe(obs, self.env)}" + SUFFIX, bool(term and reward > 0), bool(term or trunc), status

    def gold_action(self):
        """Expert bot, one instance per episode advanced with the action taken
        (a fresh bot per step oscillates on some levels)."""
        from minigrid.utils.baby_ai_bot import BabyAIBot
        if self.bot is None:
            self.bot = BabyAIBot(self.env.unwrapped)
        aid = self.bot.replan(self.prev)
        return NAMES[aid] if aid < len(NAMES) else None

    def close(self):
        self.env.close()
