# julia-mcp sessions

This file complements "Persistent Julia sessions via MCP" in
[`AGENTS.md`](../../AGENTS.md). Read it before the first call to a julia-mcp
tool in a session.

julia-mcp provides the tools `julia_eval`, `julia_restart`, and
`julia_list_sessions` (in Claude Code, e.g., `mcp__julia__julia_eval`) and
starts the Julia sessions itself.

- Always pass the **absolute path** of the `run_agents` directory as
  `env_path` (e.g., `/path/to/Trixi.jl/run_agents`). Do not call
  `Pkg.activate` in the code.
- For the KernelAbstractions.jl threading backend, pass the absolute path of
  `run_agents/kernelabstractions` as `env_path` instead; check the backend with
  `Trixi._PREFERENCE_THREADING`. If a task needs yet another environment (e.g.,
  other package versions), pass its absolute path as `env_path` instead of
  falling back to the shell. If its `Manifest.toml` was copied from
  `run_agents`, replace the relative `path = ".."` of Trixi.jl by the absolute
  path of this repository.
- The developer chooses the Julia flags of the sessions (e.g.,
  `--threads=1 --check-bounds=yes`). If a task needs other flags (e.g., the
  default bounds-checking mode to exercise `@inbounds` code paths), pass them
  via `julia_cmd` (e.g., `julia_cmd = "julia --check-bounds=auto"`). This
  replaces the session of this `env_path`; pass the same `julia_cmd` in later
  calls and verify the flags with `Base.JLOptions().check_bounds`.
- `julia_eval` returns only the output printed to `stdout` and `stderr`, not
  the value of the last expression. Use `println(...)` or `display(...)` to see
  results, e.g., `println(format([...]))` for the formatter.
- The working directory of the session is the directory passed as `env_path`
  (e.g., `run_agents/`), not the repository root. Use absolute paths, e.g.,
  wrap `@run_package_tests` in `cd("/path/to/Trixi.jl/test") do ... end` to
  run test items.
- The default `timeout` is 60 seconds, and a call exceeding its timeout **kills
  the session**. Pass a generous `timeout` (e.g., `1800`) for the first call
  (`using Trixi, TestItemRunner`) and for the first run of test items or
  elixirs, which include compilation.
- Afterward, repeat `@run_package_tests` calls with different filters in
  the same session. Revise.jl is loaded automatically (if it is installed in
  the global environment), so changes to files in `src/` are picked up without
  restarting; test files are re-read by TestItemRunner.
- Call `julia_restart` with the same `env_path` after changing type
  definitions (e.g., fields of a `struct`), which Revise cannot handle, after
  re-creating or updating `run_agents`, or if the session is in a broken state.
- `trixi_include` writes its output files to `out/` in the working directory,
  e.g., to `run_agents/out/`.
- Run the formatter (see "Code style" in `AGENTS.md`) in a separate session
  with the absolute path of `run_agents/formatter` as `env_path`, e.g.,
  `using JuliaFormatter; format(["/path/to/Trixi.jl/src/file.jl"])`, since
  JuliaFormatter.jl is not available in the `run_agents` environment. Use
  absolute paths since the working directory is `run_agents/formatter/`.
