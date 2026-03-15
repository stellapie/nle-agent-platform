"""Localization engine for translating NLE messages and names."""
import re
from typing import Optional
from localization.zh_cn import (
    MONSTER_NAMES, ITEM_NAMES, TERRAIN_NAMES,
    MESSAGE_PATTERNS, STATUS_LABELS,
)


class NLELocalizer:
    def __init__(self, lang: str = "zh_cn"):
        self.lang = lang
        self.monster_dict = MONSTER_NAMES
        self.item_dict = ITEM_NAMES
        self.terrain_dict = TERRAIN_NAMES
        self.message_patterns = MESSAGE_PATTERNS
        self.status_labels = STATUS_LABELS

    def translate_message(self, msg: str) -> str:
        if not msg:
            return msg
        result = msg
        for en, zh in self.message_patterns:
            if en in result:
                result = result.replace(en, zh)
        for en, zh in self.monster_dict.items():
            pattern = re.compile(r'\bthe\s+' + re.escape(en) + r'\b|\ba\s+' + re.escape(en) + r'\b|\b' + re.escape(en) + r'\b', re.IGNORECASE)
            result = pattern.sub(zh, result)
        return result

    def translate_entity(self, name: str) -> str:
        if not name:
            return name
        lower = name.lower().strip()
        is_pet = False
        if lower.endswith(" (pet)"):
            is_pet = True
            lower = lower[:-6]
        if lower in self.monster_dict:
            zh = self.monster_dict[lower]
            return f"{zh}(宠物)" if is_pet else zh
        if lower in self.item_dict:
            return self.item_dict[lower]
        if lower in self.terrain_dict:
            return self.terrain_dict[lower]
        for en, zh in self.terrain_dict.items():
            if en.lower() == lower:
                return zh
        for en, zh in self.monster_dict.items():
            if en in lower:
                zh_name = lower.replace(en, zh)
                return f"{zh_name}(宠物)" if is_pet else zh_name
        for en, zh in self.item_dict.items():
            if en in lower:
                return lower.replace(en, zh)
        return f"{name}(宠物)" if is_pet else name

    def translate_status_label(self, label: str) -> str:
        return self.status_labels.get(label, label)

    def localize_observation(self, text_obs: dict) -> dict:
        localized = {}
        for key, value in text_obs.items():
            if isinstance(value, str):
                localized[key] = self.translate_message(value)
            else:
                localized[key] = value
        return localized
