# Kaimon.jl sessions

This file complements "Persistent Julia sessions via MCP" in
[`AGENTS.md`](../../AGENTS.md). Read it before the first call to a Kaimon.jl
tool in a session.

Kaimon.jl connects agents to Julia sessions (REPLs) started by the developer,
who sees the code you run in their REPL. Its tools include `ping` and `ex`
(in Claude Code, e.g., `mcp__kaimon__ex`). They are only available while the
Kaimon.jl server is running.

- Call `ping` to list the connected sessions with their 8-character keys. Use
  a session only if
  `ex(e = "using Trixi; pkgdir(Trixi)", q = false, ses = "<key>")` returns the
  absolute path of the root of this repository. Otherwise (e.g., for sessions
  of other packages, clones, or worktrees, or if no session is connected), use
  julia-mcp or the shell instead.
- Always pass `q = false` (the default `q = true` does not return the result)
  and `ses = "<key>"`.
- `ex` returns the value of the last expression or the error, together with
  the output printed to `stdout` and `stderr` (e.g., test summaries and
  details of failures). However, output capture may be bypassed for code that
  loads packages via `using` or `import`. Thus, load packages in a separate
  call first, e.g., `ex(e = "using TestItemRunner", q = false, ses = "<key>")`,
  and then run test items in another call such as
  ```julia
  cd("/path/to/Trixi.jl/test") do
      @run_package_tests filter = ti -> ti.name in (
          "TreeMesh2D Advection: elixir_advection_basic.jl",
          "Unit: Spectral analysis",
      )
  end;
  ```
  ending with `;` to avoid printing the returned test set.
- Pass `max_output = 25000` (the maximum; the default is 6000 characters) when
  running test items. Longer output is truncated in the middle, keeping its
  beginning and end (including the test summary). Since a single test item
  running an elixir can print more than 10000 characters, run only a few test
  items per call and rerun failed test items individually if their details
  were truncated.
- Kaimon.jl may remove calls such as `println` from your code; use the value
  of the last expression to return results instead.
- The session belongs to the developer. Do not restart or shut it down, do not
  add or remove packages, do not start new sessions, and avoid persistent
  changes to its global state (e.g., use `cd(...) do ... end` instead of
  `cd(...)`). If the session cannot be used (e.g., missing packages, Revise.jl
  not loaded so that changes in `src/` are not picked up, or a restart is
  required after changing type definitions), ask the developer or use
  julia-mcp or the shell instead.
- Do not use the Kaimon.jl tools `run_tests` (not designed for the test items
  of Trixi.jl, may run the complete test suite) and `format_code` (see "Code
  style" in `AGENTS.md` for the formatter).
- If an evaluation takes longer than about 30 seconds (e.g., the first run of
  test items, which includes compilation), `ex` returns a job ID (`eval_id`)
  before the evaluation is finished. In this case, wait at least 30 seconds and
  call `check_eval(eval_id = "<id>")` until its status is `completed` or
  `failed`; then, the output and the result are shown.
  Do not start other evaluations in the same session while a job is running.
- If an evaluation does not produce any output for 10 minutes, Kaimon.jl
  reports a timeout, but the evaluation may still be running in the session.
- `trixi_include` writes its output files to `out/` in the working directory of
  the session (check it with `pwd()`).
