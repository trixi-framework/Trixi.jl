# CLAUDE.md

Cross-tool agent instructions live in `AGENTS.md` (single source of truth) and
are imported below. Add only genuinely Claude-Code-specific notes here, above
the import.

## Claude Code: MCP tools may be deferred

In Claude Code, MCP tools may be *deferred*: a system reminder lists only their
names (e.g., `mcp__julia__julia_eval`), and their schemas must be loaded with
`ToolSearch` (e.g., `select:mcp__julia__julia_eval,mcp__julia__julia_restart`)
before the first call. Deferred tools count as available for the order of
preference in
[Persistent Julia sessions via MCP](AGENTS.md#persistent-julia-sessions-via-mcp).
Before running any Julia code, check the tool list and the system reminders for
`mcp__kaimon__*` and `mcp__julia__*` tools. Use the Bash tool for Julia only if
neither is usable (e.g., the Kaimon.jl server failed to connect and no julia-mcp
tools are listed) or for steps that AGENTS.md assigns to the shell (the
`run_agents` setup script).

@AGENTS.md
