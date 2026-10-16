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

The recorded provider/safety regression result was `99 passed, 1 skipped`; the
skip is the optional direct DeepSeek live test. Tests were run before service
reload and made no provider requests or trading calls.

## Controlled reload acceptance record

On 2026-10-15, the dedicated LaunchAgent was reloaded after commit
`b5d7f507a94de39e468826545cd9125c6f59924d`. Old PID `93036` exited after a
LaunchAgent-targeted SIGTERM; new PID `91308` runs the committed
`tradingagents.survivor.jarvis_service --port 8765` entrypoint from this
worktree. The label reports `running`, with one service listener. No provider
request was made during reload.

Read-only status results after reload:

| Probe | Result | Latency |
| --- | --- | ---: |
| TradingAgents `GET /v1/status` | HTTP 200, PAPER ONLY, HALT true, paper opt-in false, live broker and order transport false, cycle count 0 | 63.3 ms |
| Jarvis `GET /api/trading/status` | HTTP 200, PAPER, opt-in false, live broker and order transport false | 117.6 ms |
| Jarvis `GET /api/trading/supervisor` | HTTP 200, disabled, no last cycle | 2.2 ms |
| Jarvis `GET /api/trading/supervisor/snapshot` | HTTP 200, PAPER, opt-in false, live broker and order transport false | 2202.3 ms |

Jarvis PIDs remained unchanged at `52499` (port 8000) and `75930` (port 8010).
TradingAgents still has no consistent heartbeat (`present=false`), so paper
cycle preflight remains blocked even though the service and parser checks pass.
The paper ledger passed SQLite integrity and hash-chain verification with zero
events and zero `TRADE_EXECUTED` events; its genesis head and £20 starting
balance remain unchanged. The status reports no positions and no last cycle.

HALT remains present; paper opt-in is unset/false; autonomy remains disabled;
dry run remains enabled. The launcher revalidated the captured risk limits
before startup. No HALT change, opt-in change, trading cycle, or fill occurred.
Pre- and post-reload redacted status records are kept outside the checkout in
the owner-only `.jarvis/tradingagents-paper-attestation-service` directory.

## Independent service-liveness heartbeat implementation

The prior status heartbeat was written by cycle progress/completion only. With
HALT active, that design could never bootstrap the heartbeat required by paper
preflight. The service now owns a distinct `service_heartbeat` status object,
written atomically and refreshed every five seconds by a thread started only
after the authenticated HTTP server has bound its listener. It is bound to the
service PID and a per-start random instance ID, and status validates the
instance, refresher thread, timestamp freshness, and current HALT state. An
orderly shutdown removes only that instance's file; an abrupt shutdown leaves
evidence that becomes stale and fails closed. A standalone helper cannot claim
that the HTTP service is alive.

The existing `heartbeat` remains cycle/trial data. It is not copied into
`service_heartbeat`, and last-cycle status remains sourced from the runtime
cycle journal (`last_cycle`), which is null before the first cycle. HALT is
reported independently and remains authoritative; service liveness never
implies HALT is clear or enables paper execution. The configured test HALT path
is used consistently by both service status and the liveness attestation.

The isolated heartbeat lifecycle, service status, and safety regressions pass
offline. After the authorized reload, LaunchAgent PID 2488 owns port 8765 and
authenticated `GET /v1/status` returned HTTP 200 with `service_heartbeat` present
and consistent, age about 2.4 seconds, `halt_state=HALTED`, and `service_pid=2488`.
The previous listener PID was 91308. Status remains PAPER ONLY, HALT true,
paper opt-in false, live broker/order transport false, cycle count zero, and
last cycle null. `paper.db` and `runtime.db` both pass SQLite integrity checks;
paper events and cycles remain zero. The process received the dedicated
OpenRouter provider configuration; only the key-presence boolean was inspected.
No provider request, cycle, or order was made during reload verification.

The combined `tests/survivor` regression suite passes **346 tests**. Fixture
isolation explicitly opts hermetic tests into paper execution only within their
temporary test process and pins mocked router tests to a catalog-priced model;
production safety gates and credentials are unchanged.

# One-shot paper acceptance protocol (offline implementation only)

The isolated service now exposes authenticated acceptance reservation,
authorization, result-read, and one-shot dispatch routes. A SQLite WAL/FULL
sync store atomically binds each acceptance ID to a canonical request
fingerprint. A separate `BEGIN IMMEDIATE` state transition elects one dispatch
winner across processes before it can release the service HALT. Terminal and
uncertain IDs are non-reusable; recovery changes interrupted dispatches to
`RECONCILIATION_REQUIRED` and reasserts HALT. POST is never retried after an
ambiguous outcome; status retrieval is GET-only.

Dispatch requires an explicitly PAPER-only request, the concrete PaperBroker,
both live capability attestations false, the global
`TRADINGAGENTS_SURVIVOR_PAPER_ENABLED` flag still false, current HALT true,
fresh process-bound service liveness, and autonomous execution disabled. Only
the accepted call receives an in-memory paper permit. The service records
actual RiskEngine/report fields and appended PaperLedger IDs/hashes; it does
not synthesize fills. It restores and verifies HALT and the disabled global
opt-in in a `finally` path. A failed restoration or uncertain cycle remains
durably reconciliation-required. This protocol is not a generic paper-run
switch and is not called by the autonomous supervisor or ordinary run route.

This implementation has only been exercised offline with temporary stores and
injected cycle functions. No provider call, runtime acceptance request, cycle,
or order was run. The currently running service has not been reloaded with
these changes. Before any future runtime use, the operator must separately
provision/configure the service's authenticated Jarvis bridge and the distinct
Jarvis operator authorization privately, deploy both matching commits, and
perform a new read-only preflight. Never clear a running HALT or enable
runtime paper execution as part of deployment or health checking.

Before recording a completed acceptance, the service validates nonnegative
integer risk/accounting counters, requires `risk_decisions == approved +
rejected`, and reconciles `paper_trades` against newly appended
`TRADE_EXECUTED` events in the verified PaperLedger hash chain. Invalid or
contradictory report data is durably marked reconciliation-required and cannot
be replayed under the same acceptance ID.

The authenticated Jarvis authorization request includes a configured operator
identity. TradingAgents validates its format and persists it with the
time-limited permit through dispatch and reconciliation. The ID is an audit
principal supplied by the separately authenticated Jarvis controller; the
operator token itself is not transmitted to TradingAgents. Jarvis requires
that token and the configured identity independently at its HTTP endpoint,
then uses the existing bridge bearer token for service-to-service requests.
Jarvis also compares the remote reservation fingerprint with its own before
requesting authorization.

## Runtime status contract repair (2026-10-15)

The authenticated `GET /v1/status` returns two independent heartbeat objects:
`service_heartbeat` is process-bound liveness refreshed by the active HTTP
service and includes process identity/current HALT evidence; `heartbeat` and
`cycle_heartbeat` describe actual cycle/trial activity. A zero-cycle service
has no cycle timestamp, and the service does not synthesize one.

Deployment readiness may use a fresh, consistent `service_heartbeat` for a
halted zero-cycle service. It must still verify the process identity and all
other deployment safety controls. This is not execution authorization; the
existing HALT, opt-in, broker/transport, risk, and one-call gates remain
unchanged.

Within the paper status object, `risk_state` reports PaperBroker risk limits.
`trading_state` reports effective execution state, so HALT takes precedence
over an otherwise ACTIVE risk state. `execution_eligible` is false whenever
HALT is active, paper opt-in is off, capability evidence is missing, service
liveness is invalid, risk state blocks, or autonomous execution is enabled.

The isolated offline regression on 2026-10-15 passed **78 focused tests** and
the full TradingAgents suite passed **1,059 tests, 2 skipped, and 71
subtests**. The live, authenticated pre-reload status check returned HTTP 200
in 56.3 ms for PID 2488: PAPER ONLY, HALT active, paper opt-in false, both
live broker/transport attestations false, fresh process heartbeat, absent
cycle heartbeat, zero cycles, and no last-cycle timestamp. The old loaded code
still returned `paper.trading_state=ACTIVE`; this is the PaperBroker risk
state and was not treated as execution authorization. The new status contract
reports this as `risk_state=ACTIVE`, effective `trading_state=HALTED`, and
`execution_eligible=false` while HALT is active. No provider request or cycle
was made during these checks.

## Controlled LaunchAgent reload (2026-10-15)

After commit `c49fc180240ce6911c22bec85033293c02134835` was pushed to the
existing isolated feature branch, `com.tradingagents.survivor-paper` was
reloaded with `launchctl kickstart -k`. The old PID 2488 exited and PID 44939
now owns `127.0.0.1:8765`, running
`/Users/cc/tradingagents-paper-attestation/.venv/bin/python -m
tradingagents.survivor.jarvis_service --port 8765` from this worktree. A
Python 3.12 environment probe resolves `tradingagents` to this checkout. The
dedicated LaunchAgent and rollback snapshot remain separate from Jarvis and
the original dirty TradingAgents checkout.

The authenticated read-only `/v1/status` returned HTTP 200 in 83.7 ms. It
reported PAPER ONLY; HALT active; paper opt-in false; autonomy false; concrete
PaperBroker; both live broker and live order transport false; a fresh,
consistent service heartbeat bound to PID 44939; no cycle heartbeat, cycles,
or last-cycle timestamp; and `risk_state=ACTIVE`, effective
`trading_state=HALTED`, `execution_eligible=false`. The PaperLedger and
runtime databases passed SQLite integrity; its hash chain was valid with zero
paper events/fills and zero runtime cycles. The active HALT file was unchanged.
The `paper-acceptance.db` passed SQLite integrity with zero reservations and
zero incomplete acceptances. Risk-limit values were unchanged. This was
status verification only: no provider request, cycle, order, fill, or ledger
write occurred.

## Long-cycle acceptance acknowledgement and outcome (2026-10-16)

The one authorized runtime attempt has been consumed under acceptance ID
`paper-runtime-b1f5a5a058a64aec8cf4532d5e5686c1`. The local durable cycle
record `736a207e0a5a` is `COMPLETED` with reason `NO_PROPOSAL`: three research
attempts, zero successful research outputs, three inference failures, zero
risk decisions, zero executions, and one pence total provider cost. The
PaperLedger contains the previously recorded initial treasury event only; its
chain verified and there are no trade events or fills. HALT remains active,
paper opt-in remains disabled, and this repair did not restart the service or
issue another cycle.

Previously, the HTTP dispatch handler held the request open until research
finished. It now commits the `DISPATCHED` boundary before starting one
background worker and promptly returns the durable state. Repeated POSTs only
read the existing record. Clients poll the authenticated acceptance GET route;
a process restart converts unfinished dispatches to
`RECONCILIATION_REQUIRED` and does not resume them. Provider failures and
ledger evidence are persisted in the terminal result. A Jarvis Mission must
still receive explicit operator review before it can be finalized.

The three research-run records classify the failures as invalid OpenRouter
model responses with no usable text or tool call. The available usage ledger
does not reconcile with the cycle summary, so exact per-call usage attribution
is unknown. No evidence identified timeouts, rate limits, or token limits.
The parser continues to reject empty or unusable model output; no provider
response was requested during this repair.

Authenticated live status was not re-read during this pass because a protected
process snapshot was accidentally surfaced while locating the credential
source. Do not reuse that credential; rotate it through the approved secret
path before future authenticated reconciliation. The completed remote record
is therefore based on the earlier authenticated acceptance read plus current
local durable data, and fresh live reconciliation remains outstanding.

Verification on this code: `.venv/bin/python -m pytest -q` — **1,060 passed,
2 skipped, 18 provider-model warnings, and 71 subtests passed**. Focused
acceptance/store/content-normalization tests — **21 passed**. No provider
request or service restart was made.
# Security recovery and isolated deployment verification (2026-10-17)

The exposed value was `SURVIVOR_JARVIS_TOKEN`, the local bearer credential for
Jarvis-to-TradingAgents requests. It was replaced in the protected process
snapshot, the isolated Jarvis service environment, and the private stack
environment. Each file remains owner-only (`0600`); the new values match, and
the old token is rejected by the reloaded service with HTTP 401. The dedicated
OpenRouter key was not present in the exposed process snapshot and remains in
its existing protected `.env`. No credential value or environment snapshot is
part of this repository.

The existing LaunchAgent and rollback procedure were used. The service moved
from PID `44939` to PID `48800` on `127.0.0.1:8765`, running this checkout with
Python `3.12.13` at commit
`2d99d9d185d2903281224b64f08bc5505a1e2cdc`. Authenticated read-only
`/v1/status` returned HTTP 200 for the replacement credential and HTTP 401 for
the retired credential. Its process-bound liveness heartbeat was fresh,
consistent, and bound to PID `48800`; the separate cycle heartbeat remained
at `2026-10-16T14:50:15.997267+00:00`.

Post-reload safety state: PAPER ONLY; concrete PaperBroker; HALT active;
`TRADINGAGENTS_SURVIVOR_PAPER_ENABLED` false; autonomy false; dry-run true;
live broker unavailable; live order transport unavailable; risk limits match
the protected startup snapshot. The launcher `--check` performed no provider
request. The private runtime databases passed SQLite integrity checks; the
PaperLedger hash chain is valid with one `TREASURY_INITIALIZED` event and no
fills. One historical runtime cycle is recorded, and no cycle was added during
reload.

The historical acceptance remains `COMPLETED` in the remote reservation
store, but its result still says `cycle_status=ACTIVE` while the independent
runtime database records cycle `736a207e0a5a` as `COMPLETED/NO_PROPOSAL`. The
cycle contains three failed research runs (`InvalidModelResponseError`), zero
risk decisions, and zero executions. The remote result lacks durable
`dispatch_performed`, `halt_restored`, and dispatch-count fields; its only
ledger reference is treasury initialization. Therefore Jarvis's canonical
Mission remains `RECONCILIATION_REQUIRED`. It has not been promoted to
success, and exact HTTP dispatch count remains unproven pending operator
review.

The persisted model failures indicate invalid or unusable response content;
the available metadata does not establish a timeout, rate limit, or token
limit. One usage record reports successful provider transport despite the
research run failing validation, so HTTP/provider success is not treated as a
valid trading decision. Existing parsing rejects unusable output. No provider
request or trading cycle was made during reload.

Regression verification: focused provider/acceptance/bridge/execution-boundary
tests `55 passed`; full TradingAgents suite `1060 passed, 2 skipped, 18
warnings, 71 subtests passed`. Jarvis and its full regression results are recorded in its
isolated acceptance documentation. Jarvis port `8010` reloaded from PID `45493`
to `49056`; main Jarvis port `8000` stayed at PID `52499`. No order or fill
occurred. HALT remains active and paper opt-in remains disabled.
