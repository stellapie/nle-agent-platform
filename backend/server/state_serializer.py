import numpy as np
from typing import Optional


BLSTAT_FIELDS = [
    "strength_percentage", "strength", "dexterity", "constitution",
    "intelligence", "wisdom", "charisma", "score", "hp", "hp_max",
    "depth", "gold", "energy", "energy_max", "ac", "monster_level",
    "experience_level", "experience_points", "time", "hunger_state",
    "carrying_capacity", "dungeon_number", "level_number",
    "prop_mask", "alignment", "position_x", "position_y",
]


class StateSerializer:
    def __init__(self):
        self._prev_glyphs: Optional[np.ndarray] = None
        self._prev_chars: Optional[np.ndarray] = None
        self._prev_colors: Optional[np.ndarray] = None
        self._step = 0

    def serialize_full(self, obs: dict, reward: float = 0.0, done: bool = False) -> dict:
        self._step += 1
        self._prev_glyphs = obs["glyphs"].copy()
        self._prev_chars = obs["chars"].copy()
        self._prev_colors = obs["colors"].copy()

        return {
            "type": "game_state",
            "step": self._step,
            "data": {
                "glyphs": obs["glyphs"].tolist(),
                "chars": obs["chars"].tolist(),
                "colors": obs["colors"].tolist(),
                "specials": obs["specials"].tolist(),
                "blstats": self._parse_blstats(obs["blstats"]),
                "message": self._decode_message(obs),
                "inventory": self._parse_inventory(obs),
                "tty_chars": obs["tty_chars"].tolist(),
                "tty_colors": obs["tty_colors"].tolist(),
                "tty_cursor": obs["tty_cursor"].tolist() if "tty_cursor" in obs else [0, 0],
                "done": done,
                "reward": float(reward),
            },
        }

    def serialize_delta(self, obs: dict, reward: float = 0.0, done: bool = False) -> dict:
        self._step += 1
        glyphs = obs["glyphs"]
        chars = obs["chars"]
        colors = obs["colors"]

        changes = {}
        cells = []
        if self._prev_glyphs is not None:
            diff_mask = (glyphs != self._prev_glyphs) | (chars != self._prev_chars) | (colors != self._prev_colors)
            ys, xs = np.where(diff_mask)
            for y, x in zip(ys.tolist(), xs.tolist()):
                cells.append({
                    "x": x, "y": y,
                    "glyph": int(glyphs[y, x]),
                    "char": int(chars[y, x]),
                    "color": int(colors[y, x]),
                })

        changes["cells"] = cells
        changes["blstats"] = self._parse_blstats(obs["blstats"])
        changes["message"] = self._decode_message(obs)
        changes["inventory"] = self._parse_inventory(obs)
        changes["done"] = done
        changes["reward"] = float(reward)

        self._prev_glyphs = glyphs.copy()
        self._prev_chars = chars.copy()
        self._prev_colors = colors.copy()

        return {
            "type": "game_state_delta",
            "step": self._step,
            "base_step": self._step - 1,
            "changes": changes,
        }

    def _parse_blstats(self, blstats: np.ndarray) -> dict:
        result = {}
        for i, name in enumerate(BLSTAT_FIELDS):
            if i < len(blstats):
                result[name] = int(blstats[i])
        return result

    def _decode_message(self, obs: dict) -> str:
        msg = obs.get("message", b"")
        if isinstance(msg, np.ndarray):
            return bytes(msg).decode("ascii", errors="ignore").strip("\x00").strip()
        if isinstance(msg, bytes):
            return msg.decode("ascii", errors="ignore").strip()
        return str(msg)

    def _parse_inventory(self, obs: dict) -> list:
        inv_strs = obs.get("inv_strs", [])
        inv_letters = obs.get("inv_letters", [])
        inv_glyphs = obs.get("inv_glyphs", [])
        inv_oclasses = obs.get("inv_oclasses", [])

        items = []
        for i in range(len(inv_letters)):
            letter = int(inv_letters[i]) if i < len(inv_letters) else 0
            if letter == 0:
                break
            desc = ""
            if i < len(inv_strs):
                if isinstance(inv_strs[i], np.ndarray):
                    desc = bytes(inv_strs[i]).decode("ascii", errors="ignore").strip("\x00").strip()
                else:
                    desc = str(inv_strs[i])
            glyph = int(inv_glyphs[i]) if i < len(inv_glyphs) else 0
            oclass = int(inv_oclasses[i]) if i < len(inv_oclasses) else 0
            items.append({
                "letter": chr(letter),
                "glyph": glyph,
                "class": oclass,
                "description": desc,
            })
        return items

    def reset(self):
        self._prev_glyphs = None
        self._prev_chars = None
        self._prev_colors = None
        self._step = 0
