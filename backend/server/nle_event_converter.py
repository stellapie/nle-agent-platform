"""Convert NLE observations into nethack-3d RuntimeEvent format."""
import numpy as np
from typing import Optional

BL_FIELD_MAP_367 = {
    0: "BL_TITLE", 1: "BL_STR", 2: "BL_DX", 3: "BL_CO",
    4: "BL_IN", 5: "BL_WI", 6: "BL_CH", 7: "BL_ALIGN",
    8: "BL_SCORE", 9: "BL_CAP", 10: "BL_GOLD",
    11: "BL_ENE", 12: "BL_ENEMAX", 13: "BL_XP", 14: "BL_AC",
    15: "BL_HD", 16: "BL_TIME", 17: "BL_HUNGER",
    18: "BL_HP", 19: "BL_HPMAX", 20: "BL_LEVELDESC",
    21: "BL_EXP", 22: "BL_CONDITION",
}

BLSTAT_TO_BL = {
    1: (1, "BL_STR", "number"),
    2: (2, "BL_DX", "number"),
    3: (3, "BL_CO", "number"),
    4: (4, "BL_IN", "number"),
    5: (5, "BL_WI", "number"),
    6: (6, "BL_CH", "number"),
    7: (8, "BL_SCORE", "number"),
    8: (18, "BL_HP", "number"),
    9: (19, "BL_HPMAX", "number"),
    10: (10, "BL_GOLD", "number"),
    12: (11, "BL_ENE", "number"),
    13: (12, "BL_ENEMAX", "number"),
    14: (14, "BL_AC", "number"),
    16: (13, "BL_XP", "number"),
    17: (21, "BL_EXP", "number"),
    18: (16, "BL_TIME", "number"),
    19: (17, "BL_HUNGER", "number"),
    20: (9, "BL_CAP", "number"),
    21: (20, "BL_LEVELDESC", "string"),
    22: (22, "BL_CONDITION", "number"),
    24: (7, "BL_ALIGN", "string"),
}

ALIGN_NAMES = {-1: "Chaotic", 0: "Neutral", 1: "Lawful"}
HUNGER_NAMES = {0: "Satiated", 1: "Not Hungry", 2: "Hungry", 3: "Weak", 4: "Fainting"}
CAP_NAMES = {0: "Unencumbered", 1: "Burdened", 2: "Stressed", 3: "Strained", 4: "Overtaxed", 5: "Overloaded"}


class NLEEventConverter:
    """Translate NLE observations into RuntimeEvent dicts for nethack-3d."""

    def __init__(self):
        self._prev_glyphs: Optional[np.ndarray] = None
        self._prev_blstats: Optional[np.ndarray] = None

    def obs_to_events(self, obs: dict, reward: float = 0.0,
                      done: bool = False, full: bool = False) -> list[dict]:
        events: list[dict] = []

        if full:
            events.append({"type": "runtime_event", "event": {"type": "clear_scene"}})

        events.extend(self._map_glyph_events(obs, full))
        events.extend(self._player_position_event(obs))
        events.extend(self._status_events(obs, full))
        events.extend(self._message_events(obs))
        events.extend(self._inventory_events(obs))

        self._prev_glyphs = obs["glyphs"].copy()
        self._prev_blstats = obs["blstats"].copy()
        return events

    def _map_glyph_events(self, obs: dict, full: bool) -> list[dict]:
        glyphs = obs["glyphs"]
        chars = obs["chars"]
        colors = obs["colors"]

        tiles = []
        if full or self._prev_glyphs is None:
            for y in range(glyphs.shape[0]):
                for x in range(glyphs.shape[1]):
                    g = int(glyphs[y, x])
                    if g == 0:
                        continue
                    tiles.append({
                        "x": x, "y": y,
                        "glyph": g,
                        "char": chr(int(chars[y, x])) if int(chars[y, x]) > 31 else " ",
                        "color": int(colors[y, x]),
                        "tileIndex": g,
                        "window": 2,
                    })
        else:
            diff = (glyphs != self._prev_glyphs)
            ys, xs = np.where(diff)
            for y_val, x_val in zip(ys.tolist(), xs.tolist()):
                g = int(glyphs[y_val, x_val])
                tiles.append({
                    "x": x_val, "y": y_val,
                    "glyph": g,
                    "char": chr(int(chars[y_val, x_val])) if int(chars[y_val, x_val]) > 31 else " ",
                    "color": int(colors[y_val, x_val]),
                    "tileIndex": g,
                    "window": 2,
                })

        if tiles:
            return [{"type": "runtime_event", "event": {
                "type": "map_glyph_batch", "tiles": tiles,
            }}]
        return []

    def _player_position_event(self, obs: dict) -> list[dict]:
        bl = obs["blstats"]
        px, py = int(bl[25]), int(bl[26])
        return [{"type": "runtime_event", "event": {
            "type": "player_position", "x": px, "y": py,
        }}]

    def _status_events(self, obs: dict, full: bool) -> list[dict]:
        bl = obs["blstats"]
        events = []

        for bl_idx, (field_id, field_name, val_type) in BLSTAT_TO_BL.items():
            if bl_idx >= len(bl):
                continue
            raw = int(bl[bl_idx])
            if not full and self._prev_blstats is not None:
                if bl_idx < len(self._prev_blstats) and int(self._prev_blstats[bl_idx]) == raw:
                    continue

            if field_name == "BL_ALIGN":
                value = ALIGN_NAMES.get(raw, str(raw))
            elif field_name == "BL_HUNGER":
                value = HUNGER_NAMES.get(raw, str(raw))
            elif field_name == "BL_CAP":
                value = CAP_NAMES.get(raw, str(raw))
            elif field_name == "BL_LEVELDESC":
                dn = int(bl[21]) if len(bl) > 21 else 0
                dl = int(bl[22]) if len(bl) > 22 else 0
                value = f"Dlvl:{dl}"
            else:
                value = raw

            percent = 0
            if field_name == "BL_HP":
                hp_max = int(bl[9]) if len(bl) > 9 else 1
                percent = int(100 * raw / max(hp_max, 1))

            events.append({"type": "runtime_event", "event": {
                "type": "status_update",
                "field": field_id,
                "fieldName": field_name,
                "value": value,
                "valueType": val_type,
                "chg": 0,
                "percent": percent,
                "color": 0,
                "colormask": 0,
            }})
        return events

    def _message_events(self, obs: dict) -> list[dict]:
        msg_raw = obs.get("message", b"")
        if isinstance(msg_raw, np.ndarray):
            msg = bytes(msg_raw).decode("ascii", errors="ignore").strip("\x00").strip()
        elif isinstance(msg_raw, bytes):
            msg = msg_raw.decode("ascii", errors="ignore").strip()
        else:
            msg = str(msg_raw).strip()
        if not msg:
            return []
        return [{"type": "runtime_event", "event": {
            "type": "text", "text": msg, "window": 1, "attr": 0,
        }}]

    def _inventory_events(self, obs: dict) -> list[dict]:
        inv_strs = obs.get("inv_strs", [])
        inv_letters = obs.get("inv_letters", [])
        inv_glyphs = obs.get("inv_glyphs", [])

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
            items.append({
                "text": f"{chr(letter)} - {desc}",
                "accelerator": chr(letter),
                "glyph": glyph,
            })

        if items:
            return [{"type": "runtime_event", "event": {
                "type": "inventory_update",
                "items": items,
                "window": 4,
            }}]
        return []

    def reset(self):
        self._prev_glyphs = None
        self._prev_blstats = None
