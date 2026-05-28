from __future__ import annotations

from benchmark_common.env import bool_env, float_env, int_env, overlay_env, str_env


def test_env_helpers_parse_typed_values() -> None:
    env = {
        "ENABLED": "yes",
        "COUNT": "3",
        "RATE": "0.25",
        "NAME": "run",
    }

    assert bool_env("ENABLED", env=env)
    assert int_env("COUNT", 0, env) == 3
    assert float_env("RATE", 0.0, env) == 0.25
    assert str_env("NAME", "fallback", env) == "run"


def test_overlay_env_skips_none_and_formats_bool() -> None:
    env = overlay_env({"A": None, "B": True, "C": 12}, base={"A": "kept"})

    assert env["A"] == "kept"
    assert env["B"] == "true"
    assert env["C"] == "12"
