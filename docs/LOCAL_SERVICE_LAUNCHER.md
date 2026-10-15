# Isolated TradingAgents Local Service Launcher

This configuration applies only to `/Users/cc/tradingagents-paper-attestation`
with `290176a25428af90b8f7015fe2d7341a859503e4` as its captured service
baseline. After that baseline, the launcher accepts only clean committed
descendants. It does not use or modify
the original dirty checkout at `/Users/cc/trading bot/TradingAgents` or either
Jarvis service.

## What the launcher restores

The original PID 52293 environment, arguments, runtime path, and risk-limit
values were captured from macOS process metadata into:

`/Users/cc/.jarvis/tradingagents-paper-attestation-service/original-process-52293-env.json`

The containing directory is owner-only and the snapshot is mode `0600`. It
contains secret environment values and must remain outside Git and backups
that are not access-controlled. The launcher filters only per-process Darwin
loader metadata (`main_stack`, `ptr_munge`, and related keys), which macOS
regenerates on each exec.

`scripts/tradingagents_service_launcher.py` starts the same Python module and
port from the captured checkout and environment. It reads only
`OPENROUTER_API_KEY` from the ignored `.env`, which must remain a regular file
owned by the operator with mode `0600`. It never prints or passes the key as a
command-line argument. Other `.env` variables are rejected so that the file
cannot silently override risk or execution controls.

The launcher sets the supported `openrouter` provider, endpoint
`https://openrouter.ai/api/v1`, and the single `deepseek/deepseek-v4-flash-0731:nitro`
route. Its explicit price map is `$0.05/M` input and `$0.16/M` output tokens.
This is configuration only; `--check` makes no network or provider request.

At every start the launcher fails closed unless the captured environment still
has autonomous execution explicitly disabled, dry run enabled, paper opt-in
unset/false, PAPER-only mode, the original HALT file present, and risk limits
equal to the captured values. It accepts the captured baseline or a clean
committed descendant and refuses unexpected credentials in the captured
original process environment.

## Prepared LaunchAgent

The dedicated label is `com.tradingagents.survivor-paper`, separate from
`com.jarvis.backend` and `com.jarvis.trading-supervisor`. The manifest is
`launchagents/com.tradingagents.survivor-paper.plist`; an installed copy is at
`~/Library/LaunchAgents/com.tradingagents.survivor-paper.plist`. It has
`RunAtLoad=false` and `KeepAlive=false`. It is intentionally **not bootstrapped
or loaded** while the original service is running.

## Offline validation

From the checkout:

```sh
.venv/bin/python scripts/tradingagents_service_launcher.py --check
plutil -lint launchagents/com.tradingagents.survivor-paper.plist
```

These checks do not contact OpenRouter or invoke a trading operation.

## Start after an authorized reload

1. Confirm port 8765 is owned only by the known PID 52293 service, confirm the
   HALT file is still present, and review the launcher `--check` output.
2. Stop only the known legacy PID using SIGTERM, then verify port 8765 is free.
   Do not alter the HALT file or paper opt-in.
3. Load and start the dedicated service:

   ```sh
   launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tradingagents.survivor-paper.plist"
   launchctl kickstart "gui/$(id -u)/com.tradingagents.survivor-paper"
   ```

4. Verify the service PID, executable, working directory, listener on
   `127.0.0.1:8765`, HALT, paper opt-in, autonomy flag, dry-run flag, and
   PaperBroker/live-transport attestations using bounded read-only status
   requests. Do not run a cycle as part of service startup verification.

## Roll back to the captured startup

1. Boot out only the dedicated label and verify port 8765 is free:

   ```sh
   launchctl bootout "gui/$(id -u)/com.tradingagents.survivor-paper"
   ```

2. Start the original command from the captured environment, without loading
   `.env` or applying OpenRouter overrides:

   ```sh
   (umask 077; nohup /Users/cc/tradingagents-paper-attestation/.venv/bin/python \
     /Users/cc/tradingagents-paper-attestation/scripts/tradingagents_service_launcher.py \
     --restore-original >> "$HOME/.jarvis/tradingagents-paper-attestation-service/logs/rollback.log" \
     2>&1 </dev/null &)
   ```

3. Verify the restored process owns port 8765 and the HALT, opt-in, autonomy,
   dry-run, and risk-limit values match this captured record. The rollback
   launcher does not read `.env` and does not change the snapshot.

The rollback log is created with owner-only permissions by the shell umask.
Keep the snapshot until the service configuration is retired; do not stage or
commit it.

## OpenRouter empty-response fix

The OpenRouter response probe first used a 12-token completion limit. That
budget was too small for the configured reasoning model: a later bounded
connectivity request with `max_tokens=128` returned 24 reasoning tokens and
usable text, at a reported cost of `$0.00001859`. The successful result was a
provider check only; it did not invoke TradingAgents research or trading.

`tradingagents.llm_clients.base_client.normalize_content` now raises
`InvalidModelResponseError` when the provider response is malformed or has no
usable text and no structured tool call. Reasoning-only/empty output therefore
cannot be mistaken for a completed text response. Tool-call-only messages are
preserved for the structured-output parser, which must validate their schema
before a domain decision is produced.

Offline fixture coverage is in `tests/test_llm_content_normalization.py`.
The focused provider and safety regression command is:

```sh
.venv/bin/python -m pytest -q \
  tests/test_llm_content_normalization.py \
  tests/test_provider_registry.py \
  tests/test_deepseek_reasoning.py \
  tests/survivor/test_openrouter_routing.py \
  tests/survivor/test_halt_isolation.py \
  tests/survivor/test_paper_broker.py \
  tests/survivor/test_risk_engine.py \
  tests/survivor/test_trade_ledger.py \
  tests/survivor/test_accounting_invariants.py \
  tests/survivor/test_execution_boundary.py \
  tests/survivor/test_failclosed.py
```

The recorded result before the controlled service reload was `99 passed,
1 skipped`; the skip is the optional direct DeepSeek live test. The running
service was not restarted for the provider probe or these offline tests.
