from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def bool_env(name: str, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return values.get(name, str(int(default))).strip().lower() in TRUE_VALUES


def int_env(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    values = os.environ if env is None else env
    return int(values.get(name, str(default)))


def float_env(name: str, default: float, env: Mapping[str, str] | None = None) -> float:
    values = os.environ if env is None else env
    return float(values.get(name, str(default)))


def str_env(name: str, default: str, env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get(name, default)


def overlay_env(
    overrides: Mapping[str, Any],
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            env[name] = "true" if value else "false"
        elif isinstance(value, Path):
            env[name] = str(value)
        else:
            env[name] = str(value)
    return env


def set_env_override(env: MutableMapping[str, str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        env[name] = "true" if value else "false"
    else:
        env[name] = str(value)
