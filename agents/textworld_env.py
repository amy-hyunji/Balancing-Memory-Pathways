import os
import re
import tempfile
import unicodedata

import textworld
from textworld.generator import GameOptions, compile_game, make_game

# (world_size, nb_objects, quest_length), assigned round-robin over game indices
QUEST_TRAIN = [(2, 3, 3), (2, 4, 4), (4, 4, 4), (4, 6, 6)]
QUEST_EVAL = [(6, 6, 8), (6, 8, 10), (8, 8, 12), (8, 10, 13), (8, 12, 13)]
TREASURE_EVAL = [14, 16, 18, 20, 22, 25, 28, 30]
SEED_BASE = {"train": 10000, "valid": 20000, "test": 30000}
MAX_STEPS = 100

GOAL_PREFIX = ("You are an expert TextWorld game solver. Your goal is to generate the best "
               "next action that will lead to winning the game.\n\nEnd your output sequence "
               "with an action starting with a verb. Example: open box.\n\nHere is how to "
               "win the game:\n{task}\n\nHere is your interactions so far:")
SUFFIX = "\n\nOutput the final action directly within <action> </action> tags."

INFOS = textworld.EnvInfos(feedback=True, description=True, inventory=True, admissible_commands=True,
                           facts=True, won=True, lost=True, policy_commands=True)


def game_specs(game, split, n):
    """game in {quest, treasure}; train uses easy presets, valid/test hard ones."""
    base = SEED_BASE[split]
    specs = []
    for i in range(1, n + 1):
        if game == "quest":
            w, o, q = (QUEST_TRAIN if split == "train" else QUEST_EVAL)[(i - 1) % (4 if split == "train" else 5)]
            specs.append({"env": "textworld", "game": "quest", "id": f"quest_{base + i}", "seed": base + i,
                          "world_size": w, "nb_objects": o, "quest_length": q})
        elif game == "treasure":
            specs.append({"env": "textworld", "game": "treasure", "id": f"treasure_{base + i}", "seed": base + i,
                          "level": TREASURE_EVAL[(i - 1) % len(TREASURE_EVAL)]})
        else:
            raise ValueError(game)
    return specs


def workdir():
    d = tempfile.mkdtemp(prefix="tw_")
    os.chdir(d)
    return d


def build(spec):
    """-> (game, compiled game file)."""
    if spec["game"] == "quest":
        opts = GameOptions()
        opts.grammar.only_last_action = True
        opts.nb_rooms, opts.nb_objects, opts.quest_length = spec["world_size"], spec["nb_objects"], spec["quest_length"]
        opts.seeds = spec["seed"]
        game = make_game(opts)
        return game, compile_game(game, options=opts)
    import textworld.challenges as C
    last = None
    for k in range(10):                          # high treasure levels fail to generate for some seeds
        opts = GameOptions()
        opts.grammar.only_last_action = True
        opts.seeds = spec["seed"] + k
        try:
            game = C.CHALLENGES["tw-treasure_hunter"][1]({"level": spec["level"]}, opts)
            return game, compile_game(game, options=opts)
        except Exception as e:
            last = e
    raise RuntimeError(f"{spec['id']}: generation failed: {last!r}")


def _feedback(text):
    """Room description lines and action feedback, without the banner/prompt."""
    out = []
    for m in re.finditer(r'-= (.+?) =-\n([\s\S]*?)(?=\s*>|$)', text or ""):
        out.extend(l.strip() for l in m.group(2).strip().split("\n") if l.strip())
    for m in re.finditer(r'^([^-=>\n][^\n]*?)(?=\n\s*>)', text or "", re.MULTILINE):
        a = m.group(1).strip()
        if a and a not in out:
            out.append(a)
    return "\n".join(dict.fromkeys(out))


def _room(state, game):
    for f in getattr(state, "_facts", None) or state.get("facts") or []:
        if f.name == "at" and f.arguments[0].name == "P":
            info = game.infos.get(f.arguments[1].name)
            if info is not None and info.name:
                return info.name
    return None


def normalize(s):
    s = unicodedata.normalize("NFKC", (s or "").lower().strip())
    return re.sub(r"[.!?,;]+$", "", re.sub(r"\s+", " ", s)).strip()


class TextWorldEnv:
    parse_last = True           # the last non-empty <action> (models may echo the instruction first)
    max_new_tokens = 64

    def __init__(self, spec):
        self.game, self.game_file = build(spec)
        self.env = textworld.start(self.game_file, INFOS)
        self.max_steps = MAX_STEPS

    def _obs(self):
        s = self.state
        rn = _room(s, self.game)
        return ((f"You are now in the {rn}.\n" if rn else "") + _feedback(s.get("feedback")) + "\n"
                + (s.get("inventory") or "") + "\nAdmissible commands: " + ", ".join(self.admissible()))

    def reset(self):
        self.state = self.env.reset()
        task = (self.game.objective or "").strip()
        return GOAL_PREFIX.format(task=task) + f"\n\ncurrent state: {self._obs()}" + SUFFIX

    def step(self, action):
        """Snaps the action to an admissible command (case / punctuation only);
        an invalid action is still sent and costs a step.
        -> (next user message, won, done, status)"""
        if action is None:                      # no <action> tag: a wasted turn
            action, status = "look", "no_action_tag"
        else:
            status = "invalid"
            for c in self.admissible():
                if normalize(c) == normalize(action):
                    action, status = c, "valid"
                    break
        self.state, _, done = self.env.step(action)
        return f"current state: {self._obs()}" + SUFFIX, bool(self.state.get("won")), bool(done), status

    def admissible(self):
        return list(self.state.get("admissible_commands") or [])

    def gold_action(self):
        """The engine's shortest winning policy from the current state."""
        pol = list(self.state.get("policy_commands") or [])
        return pol[0] if pol else None

    def close(self):
        self.env.close()
        stem = os.path.splitext(self.game_file)[0]
        for ext in (".z8", ".ni", ".json", ".ulx"):
            if os.path.exists(stem + ext):
                os.remove(stem + ext)
