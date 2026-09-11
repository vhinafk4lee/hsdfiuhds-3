"""TOML configuration loading with light validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Config:
    data: Dict[str, Any]
    path: Path

    @staticmethod
    def load(path: str | Path) -> "Config":
        path = Path(path)
        with path.open("rb") as fh:
            return Config(tomllib.load(fh), path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value in (None, ""):
            raise SystemExit(f"config {self.path}: missing required key '{dotted}'")
        return value

    def schema_specs(self) -> List[str]:
        specs = self.get("pow.schema")
        if not specs:
            raise SystemExit(
                f"config {self.path}: pow.schema is empty — run 'hcminer discover' and "
                "'hcminer solve-schema' first, then paste the resolved layout here"
            )
        return list(specs)
