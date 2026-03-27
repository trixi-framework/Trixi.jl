# Copilot instructions for Trixi.jl

## Big picture architecture
- Trixi.jl is a **method-of-lines PDE framework**: elixirs define a spatial `semi` object, then `semidiscretize(semi, tspan)` creates an `ODEProblem` solved by SciML integrators.
- Simulation setup is **code-first**, not config-file-first. Primary user entrypoint is `trixi_include(...)` (re-exported from `TrixiBase`), typically used with files in `examples/`.
- The top-level assembly is in `src/Trixi.jl`; include order matters. Core domains are split into `src/equations/`, `src/meshes/`, `src/solvers/`, `src/semidiscretization/`, `src/time_integration/`, and callbacks.
- `SemidiscretizationHyperbolic(mesh, equations, ic, solver; ...)` is the central composition pattern. Keep features modular by dispatching on mesh/equation/solver combinations.
- Representative elixir structure: `examples/tree_2d_dgsem/elixir_advection_basic.jl`.

## Mesh/solver boundaries you should respect
- Mesh implementations are separated in `src/meshes/meshes.jl` (Tree/Structured/Unstructured/P4est/T8code/DGMulti families).
- Solver entrypoints live in `src/solvers/solvers.jl` (`DGSEM`, `DGMulti`, parabolic solver includes).
- Prefer adding specialized methods for valid combinations instead of forcing broad generic support across all mesh types.
- Optional package features are implemented as Julia package extensions in `ext/` (e.g. `ext/TrixiCUDAExt.jl`, `ext/TrixiMakieExt.jl`).

## Callbacks architecture
- Callback code is split into `src/callbacks_step/` and `src/callbacks_stage/`, with exports and ordering defined in `src/callbacks_step/callbacks.jl`.
- Step callbacks are typically `DiscreteCallback`/`PeriodicCallback` wrappers built from methods on callback types (see `SummaryCallback`, `AnalysisCallback`, `SaveSolutionCallback`, `StepsizeCallback`).
- Preserve callback ordering in `CallbackSet`: summary/analysis/alive/save first, then AMR, then `StepsizeCallback`, then physics-specific post-step updates.
- Interval checks should use accepted steps via `integrator.stats.naccept`; final-step behavior should use `isfinished(integrator)` from `src/callbacks_step/summary.jl`.
- `SaveSolutionCallback` has both `interval` and `dt` modes (the latter via `PeriodicCallback`); keep both paths consistent when extending output.
- Trixi has a separate stage-callback pipeline for `SimpleSSPRK33(stage_callbacks = (...))` in `src/time_integration/ssprk43.jl`; implement `init_callback`/`finalize_callback` in `src/time_integration/methods_SSP.jl` for new stage callback types.
- For output extensions, prefer overload hooks like `get_element_variables!` (used by `AMRCallback`) instead of hard-coding into save routines.
- Wire callbacks in elixirs as explicit variables plus `CallbackSet(...)` (see `examples/tree_2d_dgsem/elixir_advection_basic.jl`), and test via `@test_trixi_include` overrides in `test/test_trixi.jl` and callback-focused tests.

## Developer workflows (project-specific)
- Recommended local dev loop is REPL + Revise: `julia --project=run`, then `using Revise, Trixi`.
- Run package tests via `Pkg.test("Trixi")`; CI shards tests using `TRIXI_TEST` in `test/runtests.jl`.
- For targeted local runs, include specific files such as `include("test/test_tree_2d_part1.jl")`.
- Threading/MPI behavior in tests is intentional; `test/runtests.jl` computes `TRIXI_NTHREADS` and `TRIXI_MPI_NPROCS` dynamically.
- Keep new example elixirs covered by tests (see `examples/README.md` and existing `@trixi_testset` usage in `test/test_threaded.jl`).

## Coding and testing conventions to follow
- Formatting is enforced with JuliaFormatter (`.JuliaFormatter.toml`, SciML style, 4-space indent, max line length 92).
- Use provided formatter helper: `utils/trixi-format.jl` (or `utils/pre-commit` hook flow).
- Naming conventions are strict: `CamelCase` for types/modules, `snake_case` for functions/variables, mutating functions end with `!`.
- Internal argument-order pattern matters: backend first (if present), then generally `mesh, equations, solver, cache`; modified argument first for mutating APIs.
- Existing tests frequently use `@test_trixi_include` wrappers from `test/test_trixi.jl`; prefer this style for elixir-based regression tests.

## Practical guidance for AI edits
- When adding physics/numerics, update the appropriate domain file (`equations`, `solvers`, or `semidiscretization`) and add at least one elixir-backed test in the matching `test/test_*.jl` shard.
- Preserve include organization and avoid moving files between subsystem directories unless required by design.
- Use discoverable examples from `examples/` as templates rather than inventing new setup patterns.
- If behavior is backend-specific (CUDA/MPI/weakdeps), gate it in extension files or existing backend dispatch paths instead of unconditional imports.
