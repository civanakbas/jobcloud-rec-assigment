"""Exact, case-insensitive mapping of provider values; no text extraction."""

from collections.abc import Mapping
from typing import Any


def _index(concepts: list[dict]) -> dict[str, str]:
    index = {}
    for concept in concepts:
        for value in [concept["id"], *concept["aliases"]]:
            key = value.strip().casefold()
            previous = index.setdefault(key, concept["id"])
            if previous != concept["id"]:
                raise ValueError(f"Ambiguous taxonomy alias: {value!r}")
    return index


class Taxonomy:
    def __init__(self, document: dict):
        self.occupations = _index(document["occupations"])
        self.skills = _index(document["skills"])

    def normalize(self, payload: Mapping[str, Any]) -> tuple[dict, dict]:
        raw_occupation = payload["occupation_id"]
        occupation = (
            self.occupations.get(raw_occupation.strip().casefold())
            if raw_occupation is not None
            else None
        )
        skills, unresolved_skills = set(), set()
        for raw in payload["skill_ids"]:
            mapped = self.skills.get(raw.strip().casefold())
            if mapped is None:
                unresolved_skills.add(raw)
            else:
                skills.add(mapped)
        normalized = {**payload, "occupation_id": occupation, "skill_ids": sorted(skills)}
        unresolved = {
            "occupation_id": raw_occupation if occupation is None else None,
            "skill_ids": sorted(unresolved_skills),
        }
        return normalized, unresolved
