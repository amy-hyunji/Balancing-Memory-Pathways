def make_env(spec):
    if spec["env"] == "textworld":
        from .textworld_env import TextWorldEnv
        return TextWorldEnv(spec)
    from .babyai_env import BabyAIEnv
    return BabyAIEnv(spec)


def game_specs(env, game, split, n):
    if env == "textworld":
        from .textworld_env import game_specs as f
    else:
        from .babyai_env import game_specs as f
    return f(game, split, n)
