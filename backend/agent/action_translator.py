"""Translate high-level text actions to NLE action indices."""
import nle.nethack as nethack

DIR = nethack.actions.CompassDirection
MISC = nethack.actions.MiscDirection
CMD = nethack.actions.Command

ACTION_MAP: dict[str, list] = {
    "move north": [DIR.N], "move south": [DIR.S],
    "move east": [DIR.E], "move west": [DIR.W],
    "move northeast": [DIR.NE], "move northwest": [DIR.NW],
    "move southeast": [DIR.SE], "move southwest": [DIR.SW],
    "go downstairs": [MISC.DOWN], "go upstairs": [MISC.UP],
    "attack north": [CMD.FIGHT, DIR.N], "attack south": [CMD.FIGHT, DIR.S],
    "attack east": [CMD.FIGHT, DIR.E], "attack west": [CMD.FIGHT, DIR.W],
    "attack northeast": [CMD.FIGHT, DIR.NE], "attack northwest": [CMD.FIGHT, DIR.NW],
    "attack southeast": [CMD.FIGHT, DIR.SE], "attack southwest": [CMD.FIGHT, DIR.SW],
    "search": [CMD.SEARCH], "wait": [MISC.WAIT],
    "pickup": [CMD.PICKUP], "eat": [CMD.EAT],
    "open north": [CMD.OPEN, DIR.N], "open south": [CMD.OPEN, DIR.S],
    "open east": [CMD.OPEN, DIR.E], "open west": [CMD.OPEN, DIR.W],
    "close north": [CMD.CLOSE, DIR.N], "close south": [CMD.CLOSE, DIR.S],
    "close east": [CMD.CLOSE, DIR.E], "close west": [CMD.CLOSE, DIR.W],
    "drink potion": [CMD.QUAFF], "read scroll": [CMD.READ],
    "wield weapon": [CMD.WIELD], "wear armor": [CMD.WEAR],
    "remove armor": [CMD.REMOVE], "put on": [CMD.PUTON],
    "drop": [CMD.DROP], "throw": [CMD.THROW],
    "zap": [CMD.ZAP], "apply": [CMD.APPLY],
    "look": [CMD.LOOK], "inventory": [CMD.INVENTORY],
    "pray": [CMD.PRAY],
    "kick north": [CMD.KICK, DIR.N], "kick south": [CMD.KICK, DIR.S],
    "kick east": [CMD.KICK, DIR.E], "kick west": [CMD.KICK, DIR.W],
}

ALIASES: dict[str, str] = {
    "n": "move north", "s": "move south", "e": "move east", "w": "move west",
    "ne": "move northeast", "nw": "move northwest",
    "se": "move southeast", "sw": "move southwest",
    "up": "go upstairs", "down": "go downstairs",
    ">": "go downstairs", "<": "go upstairs",
    ".": "wait", ",": "pickup",
}


class NLEActionTranslator:
    def __init__(self, env_actions=None):
        if env_actions is None:
            env_actions = nethack.actions.ACTIONS
        self._code_to_idx = {}
        for idx, act in enumerate(env_actions):
            self._code_to_idx[int(act)] = idx

    def translate(self, text_action: str) -> list[int]:
        """Return list of action *indices* suitable for env.step()."""
        text_action = text_action.strip().lower()
        text_action = ALIASES.get(text_action, text_action)
        actions = ACTION_MAP.get(text_action)
        if actions is None:
            for key, codes in ACTION_MAP.items():
                if key.startswith(text_action) or text_action.startswith(key):
                    actions = codes
                    break
        if actions is None:
            actions = [MISC.WAIT]
        indices = []
        for a in actions:
            idx = self._code_to_idx.get(int(a))
            if idx is not None:
                indices.append(idx)
        return indices if indices else [self._code_to_idx.get(int(MISC.WAIT), 0)]

    @staticmethod
    def available_actions() -> list[str]:
        return sorted(ACTION_MAP.keys())
