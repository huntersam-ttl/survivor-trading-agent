#!/usr/bin/env python3
"""Launch the isolated SURVIVOR service with its captured safe environment.

The original process environment is held in a mode-0600 file outside the
checkout. The dedicated OpenRouter key is read from the ignored, mode-0600
checkout .env and is never written to the process command line or logs.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
from dataclasses import asdict


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DOTENV = ROOT / ".env"
SNAPSHOT = pathlib.Path.home() / ".jarvis" / "tradingagents-paper-attestation-service" / "original-process-52293-env.json"
EXPECTED_BASE_COMMIT = "290176a25428af90b8f7015fe2d7341a859503e4"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash-0731:nitro"
PRICE_MAP = f"{MODEL}:0.05,0.16"
VOLATILE_DARWIN_KEYS = {
    "dyld_file", "executable_file", "main_stack", "ptr_munge", "security_config", "th_port",
}


class ConfigurationError(Exception):
    pass


def _read_snapshot() -> dict:
    try:
        info = SNAPSHOT.stat()
    except OSError as exc:
        raise ConfigurationError("private original environment snapshot is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ConfigurationError("private original environment snapshot permissions are unsafe")
    try:
        data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError("private original environment snapshot is unreadable") from exc
    if data.get("schema") != 1 or data.get("cwd") != str(ROOT):
        raise ConfigurationError("snapshot does not belong to this isolated checkout")
    if data.get("repo_commit") != EXPECTED_BASE_COMMIT:
        raise ConfigurationError("snapshot does not match the isolated service baseline")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if head != EXPECTED_BASE_COMMIT:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, "HEAD"],
                cwd=ROOT, check=True, capture_output=True,
            )
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout
            if dirty:
                raise ConfigurationError("service checkout must be clean after its baseline commit")
    except ConfigurationError:
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError("checkout is not a clean committed descendant of the service baseline") from exc
    return data


def _snapshot_environment(data: dict) -> dict[str, str]:
    items = data.get("environment")
    if not isinstance(items, list):
        raise ConfigurationError("snapshot environment is malformed")
    env: dict[str, str] = {}
    for item in items:
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(x, str) for x in item):
            raise ConfigurationError("snapshot environment is malformed")
        key, value = item
        if key in env:
            raise ConfigurationError("snapshot contains duplicate environment names")
        env[key] = value
    for key in VOLATILE_DARWIN_KEYS:
        # These are per-process dyld/kernel metadata. A new exec receives fresh
        # values from macOS; replaying captured addresses would be incorrect.
        env.pop(key, None)
    return env


def _read_dedicated_key() -> str:
    try:
        info = DOTENV.lstat()
    except OSError as exc:
        raise ConfigurationError("dedicated TradingAgents .env is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ConfigurationError("dedicated TradingAgents .env must be a regular owner-only 0600 file")
    try:
        from dotenv import dotenv_values

        values = dotenv_values(DOTENV, interpolate=False)
    except Exception as exc:
        raise ConfigurationError("dedicated TradingAgents .env could not be parsed") from exc
    configured_names = {key for key, value in values.items() if key and value is not None}
    if configured_names != {"OPENROUTER_API_KEY"}:
        raise ConfigurationError("dedicated .env must contain only OPENROUTER_API_KEY")
    key = values.get("OPENROUTER_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ConfigurationError("dedicated OpenRouter credential is empty")
    return key


def _prepared_environment(data: dict, *, include_provider: bool) -> dict[str, str]:
    env = _snapshot_environment(data)
    os.environ.clear()
    os.environ.update(env)

    # Safe controls are assertions, not values repaired by the launcher.
    if env.get("TRADINGAGENTS_SURVIVOR_AUTONOMY_ENABLED", "").strip().lower() not in {"false", "0", "no", "off"}:
        raise ConfigurationError("captured autonomous execution control is not explicitly disabled")
    if env.get("SURVIVOR_DRY_RUN", "").strip().lower() not in {"true", "1", "yes", "on"}:
        raise ConfigurationError("captured dry-run control is not explicitly enabled")
    if env.get("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", "").strip().lower() in {"true", "1", "yes", "on"}:
        raise ConfigurationError("captured paper opt-in is enabled")
    if env.get("TRADING_MODE", "").strip().upper() not in {"", "PAPER", "PAPER_ONLY", "PAPER ONLY"}:
        raise ConfigurationError("captured trading mode is not paper-only")
    if env.get("OPENROUTER_API_KEY", "").strip():
        raise ConfigurationError("original service environment already contains a provider key")

    runtime = pathlib.Path(data.get("runtime_dir", ""))
    halt_path = pathlib.Path(data.get("halt_path", ""))
    if not runtime.is_absolute() or halt_path != runtime / "HALT" or not halt_path.is_file():
        raise ConfigurationError("captured runtime HALT latch is missing or inconsistent")

    from tradingagents.survivor.risk.limits import risk_limits_from_env

    if asdict(risk_limits_from_env()) != data.get("risk_limits"):
        raise ConfigurationError("captured risk limits no longer match the running service configuration")

    if include_provider:
        env["OPENROUTER_API_KEY"] = _read_dedicated_key()
        env.update({
            "TRADINGAGENTS_LLM_PROVIDER": "openrouter",
            "TRADINGAGENTS_LLM_BACKEND_URL": OPENROUTER_URL,
            "SURVIVOR_OPENROUTER_DEFAULT_MODEL": MODEL,
            "SURVIVOR_OPENROUTER_MODELS": MODEL,
            "SURVIVOR_OPENROUTER_PRICING_MAP": PRICE_MAP,
        })
        os.environ.clear()
        os.environ.update(env)
        from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS
        from tradingagents.llm_clients.validators import validate_model

        provider = OPENAI_COMPATIBLE_PROVIDERS.get("openrouter")
        if provider is None or provider.base_url != OPENROUTER_URL:
            raise ConfigurationError("TradingAgents OpenRouter endpoint does not match the expected endpoint")
        if env.get("TRADINGAGENTS_LLM_BACKEND_URL") != OPENROUTER_URL or not validate_model("openrouter", MODEL):
            raise ConfigurationError("TradingAgents OpenRouter provider/model configuration is invalid")
    return env


def _check() -> int:
    data = _read_snapshot()
    env = _prepared_environment(data, include_provider=True)
    print("configuration=valid")
    print("provider=openrouter")
    print(f"endpoint={OPENROUTER_URL}")
    print(f"model={MODEL}")
    print("dedicated_key_present=true (value hidden)")
    print("paper_mode=PAPER ONLY (TRADING_MODE unset; project default preserved)")
    print("paper_opt_in=" + (env.get("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED") or "unset/false"))
    print("autonomy_enabled=" + env["TRADINGAGENTS_SURVIVOR_AUTONOMY_ENABLED"])
    print("dry_run=" + env["SURVIVOR_DRY_RUN"])
    print("HALT=active")
    print("risk_limits=" + json.dumps(data["risk_limits"], sort_keys=True))
    print("provider_request=not performed")
    return 0


def _exec_service(*, restore_original: bool) -> None:
    data = _read_snapshot()
    env = _prepared_environment(data, include_provider=not restore_original)
    cwd = data["cwd"]
    os.chdir(cwd)
    if restore_original:
        argv = data.get("argv")
        if not isinstance(argv, list) or not argv or argv[0] != str(PYTHON):
            raise ConfigurationError("captured original command is invalid")
        os.execve(argv[0], argv, env)
    argv = [str(PYTHON), "-m", "tradingagents.survivor.jarvis_service", "--port", "8765"]
    os.execve(str(PYTHON), argv, env)


def main() -> int:
    try:
        if sys.argv[1:] == ["--check"]:
            return _check()
        if sys.argv[1:] == ["--restore-original"]:
            _exec_service(restore_original=True)
        if sys.argv[1:]:
            raise ConfigurationError("unknown launcher option")
        _exec_service(restore_original=False)
    except ConfigurationError as exc:
        print(f"TradingAgents launcher refused startup: {exc}", file=sys.stderr)
        return 78
    except Exception as exc:
        # Never include arbitrary exception text: providers and SDKs can embed
        # request details. Type-only diagnostics are sufficient for launchd.
        print(f"TradingAgents launcher failed: {type(exc).__name__}", file=sys.stderr)
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
