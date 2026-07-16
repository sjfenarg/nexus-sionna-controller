from __future__ import annotations

import re
from collections.abc import Iterable


def unique_entity_id(prefix: str, existing_ids: Iterable[str]) -> str:
    """Return the first ``prefix<n>`` identifier not present in ``existing_ids``."""
    normalized_prefix = _normalize_prefix(prefix)
    used_indices = set()
    exact_ids = set(existing_ids)
    pattern = re.compile(rf"^{re.escape(normalized_prefix)}(\d+)$")
    for entity_id in exact_ids:
        match = pattern.match(entity_id)
        if match:
            used_indices.add(int(match.group(1)))

    index = 0
    while index in used_indices or f"{normalized_prefix}{index}" in exact_ids:
        index += 1
    return f"{normalized_prefix}{index}"


def _normalize_prefix(prefix: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", prefix.strip().lower()).strip("_")
    return normalized or "entity"
