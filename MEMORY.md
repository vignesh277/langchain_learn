# Project Memory

## Project
New folder repository.

## Stack
Unknown (Python files present in `agents/`).

## What's Been Built
- 2026-04-29: Initial memory file created.
- 2026-04-29: Patched `agents/agents_dynamic_tools.py` for safer middleware state handling and true search alias behavior.
- 2026-04-29: Fixed tool-alias runtime bug by introducing shared non-decorated search implementation used by both `public_search` and `brave_search`.
- 2026-04-29: Added safer learning/test harness for state-based tool filtering with recursion guard, system prompt guidance, and friendly rate-limit handling.
- 2026-04-29: Refactored tool-filter logic into pure function and added deterministic local tests to avoid LLM loop/rate-limit issues during learning.
- 2026-04-29: Removed optional live-run test path from `agents_dynamic_tools.py` to keep the script focused on manual/local verification.
- 2026-04-29: Restored minimal execution block to run local filter tests when script is executed directly.
- 2026-04-29: Default `__main__` runs `run_groq_examples()` so middleware tool filtering is visible on real Groq calls; `run_local_filter_tests()` kept for optional offline checks.

## Current State
- Git repo with untracked files in `agents/`.
- `agents/agents_dynamic_tools.py` now uses safe `messages` lookup and `brave_search` delegates to `public_search`.
- `brave_search` no longer calls a `StructuredTool` object directly, avoiding `TypeError: 'StructuredTool' object is not callable`.
- `agents_dynamic_tools.py` `__main__` runs three Groq scenarios with `recursion_limit=12`; middleware prints `authenticated`, `messages_count`, and `tools_selected` on each model step. `run_local_filter_tests()` remains for offline policy checks.

## Key Files & Locations
- `agents/agents_dynamic_tools.py`
- `agents/agents_dynamicmodel.py`
- `agents/agents_static_tools.py`
- `agents/agents_staticmodel.py`

## Credentials & Config
- None recorded.

## Decisions Made
- Use `MEMORY.md` for persistent project context.

## What's Next
- Review and validate current agent files as requested.

## Change Log
- 2026-04-29 | Created `MEMORY.md` with initial project snapshot.
- 2026-04-29 | Updated `agents/agents_dynamic_tools.py` to avoid missing-state crashes and fix `brave_search` alias return.
- 2026-04-29 | Reworked search alias implementation to use shared helper function so both tools return equivalent output safely.
- 2026-04-29 | Added loop-safe invocation wrapper and model guidance prompt for LangChain state-based tool filtering tests.
- 2026-04-29 | Added deterministic middleware filter tests and moved live agent calls behind environment flag.
- 2026-04-29 | Removed live run helper and env-flag branch per user preference for simpler learning flow.
- 2026-04-29 | Re-added minimal main execution block for local deterministic test output.
- 2026-04-29 | Switched default `__main__` to `run_groq_examples()` for live Groq invocations with visible middleware logs.
- 2026-04-29 | Added `AgentStateWithAuth` + `state_schema=` so `authenticated` from `invoke` reaches middleware (fixes always-False auth in logs).
