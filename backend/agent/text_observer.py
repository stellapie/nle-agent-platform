"""Convert NLE raw observations into structured text for LLM consumption."""
import numpy as np
from typing import Optional
import nle.nethack as nethack

GLYPH_MON_OFF = nethack.GLYPH_MON_OFF
GLYPH_PET_OFF = nethack.GLYPH_PET_OFF
GLYPH_INVIS_OFF = nethack.GLYPH_INVIS_OFF
GLYPH_DETECT_OFF = nethack.GLYPH_DETECT_OFF
GLYPH_BODY_OFF = nethack.GLYPH_BODY_OFF
GLYPH_RIDDEN_OFF = nethack.GLYPH_RIDDEN_OFF
GLYPH_OBJ_OFF = nethack.GLYPH_OBJ_OFF
GLYPH_CMAP_OFF = nethack.GLYPH_CMAP_OFF
GLYPH_EXPLODE_OFF = nethack.GLYPH_EXPLODE_OFF
GLYPH_ZAP_OFF = nethack.GLYPH_ZAP_OFF
GLYPH_SWALLOW_OFF = nethack.GLYPH_SWALLOW_OFF
GLYPH_WARNING_OFF = nethack.GLYPH_WARNING_OFF
GLYPH_STATUE_OFF = nethack.GLYPH_STATUE_OFF
NUMMONS = nethack.NUMMONS

DIRECTION_NAMES = [
    ("north", 0, -1), ("south", 0, 1),
    ("east", 1, 0), ("west", -1, 0),
    ("northeast", 1, -1), ("northwest", -1, -1),
    ("southeast", 1, 1), ("southwest", -1, 1),
]

ALIGN_NAMES = {-1: "Chaotic", 0: "Neutral", 1: "Lawful"}
HUNGER_NAMES = {0: "Satiated", 1: "Not Hungry", 2: "Hungry", 3: "Weak", 4: "Fainting", 5: "Fainted", 6: "Starved"}


def glyph_to_name(glyph: int) -> Optional[str]:
    """Best-effort glyph-to-name resolution."""
    if GLYPH_MON_OFF <= glyph < GLYPH_MON_OFF + NUMMONS:
        try:
            mon = nethack.permonst(glyph - GLYPH_MON_OFF)
            return mon.mname
        except Exception:
            return f"monster_{glyph}"
    if GLYPH_PET_OFF <= glyph < GLYPH_PET_OFF + NUMMONS:
        try:
            mon = nethack.permonst(glyph - GLYPH_PET_OFF)
            return f"{mon.mname} (pet)"
        except Exception:
            return f"pet_{glyph}"
    if GLYPH_OBJ_OFF <= glyph < GLYPH_CMAP_OFF:
        try:
            obj = nethack.objclass(glyph - GLYPH_OBJ_OFF)
            return obj.oc_name if hasattr(obj, 'oc_name') else f"object_{glyph}"
        except Exception:
            return f"object_{glyph}"
    if GLYPH_CMAP_OFF <= glyph < GLYPH_EXPLODE_OFF:
        cmap_id = glyph - GLYPH_CMAP_OFF
        CMAP_NAMES = {
            0: "dark area", 1: "stone", 2: "vertical wall", 3: "horizontal wall",
            4: "top-left corner", 5: "top-right corner", 6: "bottom-left corner",
            7: "bottom-right corner", 8: "cross wall", 9: "upward T-wall",
            10: "downward T-wall", 11: "leftward T-wall", 12: "rightward T-wall",
            13: "open door", 14: "closed door", 15: "iron bars",
            16: "tree", 17: "floor", 18: "corridor", 19: "lit corridor",
            20: "staircase up", 21: "staircase down", 22: "ladder up",
            23: "ladder down", 24: "altar", 25: "grave", 26: "throne",
            27: "sink", 28: "fountain", 29: "pool", 30: "ice",
            31: "lava", 32: "lowered drawbridge", 33: "raised drawbridge",
        }
        return CMAP_NAMES.get(cmap_id, f"terrain_{cmap_id}")
    return None


class NLETextObserver:
    """Produce structured text observations from NLE numpy arrays."""

    def observe(self, obs: dict, meta: Optional[dict] = None) -> dict:
        return {
            "surroundings": self._describe_surroundings(obs),
            "status": self._describe_status(obs, meta),
            "message": self._decode_message(obs),
            "inventory": self._describe_inventory(obs),
            "adjacent": self._describe_adjacent(obs),
            "visible_entities": self._list_visible_entities(obs),
            "meta": self._describe_meta(meta) if meta else "",
        }

    def observe_as_prompt(self, obs: dict, meta: Optional[dict] = None) -> str:
        parts = self.observe(obs, meta)
        sections = []
        if parts["message"]:
            sections.append(f"[Message] {parts['message']}")
        sections.append(f"[Status] {parts['status']}")
        sections.append(f"[Surroundings] {parts['surroundings']}")
        if parts["adjacent"]:
            sections.append(f"[Adjacent] {parts['adjacent']}")
        if parts["visible_entities"]:
            sections.append(f"[Visible Entities] {parts['visible_entities']}")
        if parts["inventory"]:
            sections.append(f"[Inventory] {parts['inventory']}")
        if parts["meta"]:
            sections.append(f"[Score System] {parts['meta']}")
        return "\n".join(sections)

    def _describe_surroundings(self, obs: dict) -> str:
        bl = obs["blstats"]
        px, py = int(bl[25]), int(bl[26])
        glyphs = obs["glyphs"]
        chars = obs["chars"]

        standing_on = glyph_to_name(int(glyphs[py, px]))
        dl = int(bl[22]) if len(bl) > 22 else 1
        parts = [f"Dungeon level {dl}."]
        if standing_on:
            parts.append(f"Standing on: {standing_on}.")
        return " ".join(parts)

    def _describe_adjacent(self, obs: dict) -> str:
        bl = obs["blstats"]
        px, py = int(bl[25]), int(bl[26])
        glyphs = obs["glyphs"]
        h, w = glyphs.shape
        adj = []
        for name, dx, dy in DIRECTION_NAMES:
            nx, ny = px + dx, py + dy
            if 0 <= ny < h and 0 <= nx < w:
                g = int(glyphs[ny, nx])
                tile_name = glyph_to_name(g)
                if tile_name and tile_name not in ("dark area", "stone"):
                    adj.append(f"{name}: {tile_name}")
        return ", ".join(adj) if adj else ""

    def _list_visible_entities(self, obs: dict) -> str:
        bl = obs["blstats"]
        px, py = int(bl[25]), int(bl[26])
        glyphs = obs["glyphs"]
        h, w = glyphs.shape
        entities = []
        for y in range(max(0, py - 10), min(h, py + 11)):
            for x in range(max(0, px - 10), min(w, px + 11)):
                if x == px and y == py:
                    continue
                g = int(glyphs[y, x])
                is_monster = GLYPH_MON_OFF <= g < GLYPH_MON_OFF + NUMMONS
                is_pet = GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS
                if is_monster or is_pet:
                    name = glyph_to_name(g)
                    dx, dy_val = x - px, y - py
                    dist = max(abs(dx), abs(dy_val))
                    direction = self._offset_to_direction(dx, dy_val)
                    entities.append(f"{name} ({direction}, {dist} tiles)")
        return ", ".join(entities) if entities else ""

    def _offset_to_direction(self, dx: int, dy: int) -> str:
        parts = []
        if dy < 0:
            parts.append("north")
        elif dy > 0:
            parts.append("south")
        if dx > 0:
            parts.append("east")
        elif dx < 0:
            parts.append("west")
        return "".join(parts) or "here"

    def _describe_status(self, obs: dict, meta: Optional[dict]) -> str:
        bl = obs["blstats"]
        hp = int(bl[8]) if len(bl) > 8 else 0
        hp_max = int(bl[9]) if len(bl) > 9 else 0
        mp = int(bl[12]) if len(bl) > 12 else 0
        mp_max = int(bl[13]) if len(bl) > 13 else 0
        ac = int(bl[14]) if len(bl) > 14 else 0
        lvl = int(bl[16]) if len(bl) > 16 else 0
        xp = int(bl[17]) if len(bl) > 17 else 0
        gold = int(bl[11]) if len(bl) > 11 else 0
        st = int(bl[1]) if len(bl) > 1 else 0
        dx_val = int(bl[2]) if len(bl) > 2 else 0
        co = int(bl[3]) if len(bl) > 3 else 0
        in_val = int(bl[4]) if len(bl) > 4 else 0
        wi = int(bl[5]) if len(bl) > 5 else 0
        ch = int(bl[6]) if len(bl) > 6 else 0
        hunger = int(bl[19]) if len(bl) > 19 else 1
        align = int(bl[24]) if len(bl) > 24 else 0

        hunger_str = HUNGER_NAMES.get(hunger, str(hunger))
        align_str = ALIGN_NAMES.get(align, str(align))

        line = (f"HP:{hp}/{hp_max} MP:{mp}/{mp_max} AC:{ac} "
                f"Lv:{lvl} XP:{xp} Gold:{gold} | "
                f"St:{st} Dx:{dx_val} Co:{co} In:{in_val} Wi:{wi} Ch:{ch} | "
                f"Hunger:{hunger_str} Align:{align_str}")
        return line

    def _describe_inventory(self, obs: dict) -> str:
        inv_strs = obs.get("inv_strs", [])
        inv_letters = obs.get("inv_letters", [])
        items = []
        for i in range(len(inv_letters)):
            letter = int(inv_letters[i])
            if letter == 0:
                break
            desc = ""
            if i < len(inv_strs):
                if isinstance(inv_strs[i], np.ndarray):
                    desc = bytes(inv_strs[i]).decode("ascii", errors="ignore").strip("\x00").strip()
                else:
                    desc = str(inv_strs[i])
            items.append(f"{chr(letter)}: {desc}")
        return "; ".join(items) if items else "empty"

    def _decode_message(self, obs: dict) -> str:
        msg = obs.get("message", b"")
        if isinstance(msg, np.ndarray):
            return bytes(msg).decode("ascii", errors="ignore").strip("\x00").strip()
        if isinstance(msg, bytes):
            return msg.decode("ascii", errors="ignore").strip()
        return str(msg).strip()

    def _describe_meta(self, meta: Optional[dict]) -> str:
        if not meta:
            return ""
        ts = meta.get("total_score", 0)
        deaths = meta.get("deaths", 0)
        steps = meta.get("total_steps", 0)
        ep = meta.get("episode_count", 0)
        return f"Score:{ts:.0f} Deaths:{deaths} Steps:{steps} Episode:{ep}"
