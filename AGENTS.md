# AGENTS.md

Canonical agent instructions for Trixi.jl. Read natively by Codex (and Cursor,
Gemini CLI, Aider, …). Claude Code reads it via the `@AGENTS.md` import in
`CLAUDE.md`. Keep this file the single source of truth; tool-specific notes go in
the respective tool's file.

Trixi.jl is a Julia library for adaptive high-order numerical simulations of
hyperbolic PDEs (and related parabolic/multi-physics problems). Authoritative
developer docs live in `docs/src/` — this file summarizes the parts agents need
most and links back for detail.

## Build, test, format

- Full test suite: Never attempt to run the full test suite locally. It takes much too long (multiple hours).
- To `include` a single test file, start Julia from the root directory (`julia --project=.`) and use TestEnv.jl, e.g.,
  ```julia
  using TestEnv; TestEnv.activate()
  include(joinpath("test", "test_tree_1d_euler.jl")
  ```
  Files are named `test_<mesh>_<dim>_<eq>.jl`; CI splits them into parallel jobs selected by the
  `TRIXI_TEST` variable in `test/runtests.jl`.
- Examples live in `examples/` and development scripts often live in `run/`. They depend on packages **not** in the main
  `Project.toml` (Plots, Makie, OrdinaryDiffEqXYZ, ...). If present, use the dedicated `run/` project,
  which `dev`s the local package and has those extras installed, e.g., to run an example already in the `examples/` folder:
   ```sh
  julia --project=run examples/tree_1d_dgsem/elixir_euler_ec.jl
  ```
- For both when running tests and when running examples, if available use the MCP julia tool
  `mcp__julia__julia_eval` to keep a persistent julia session. For running examples, `env_path` should
  point at `run/` and for running tests `env_path` should point at `.`.
- Format before committing (CI enforces a no-op): `utils/trixi-format.jl` or
  `julia -e 'using JuliaFormatter; format(".")'` (SciML style, `.JuliaFormatter.toml`).

## Repository layout

- `src/` — library code (walked in **Architecture** below).
- `examples/` — elixirs.
- `test/` — test suite (entry point `runtests.jl`).
- `ext/` — weak-dep extensions (Makie, Plots, CUDA, AMDGPU, …).
- `docs/` — documentation source.
- `utils/` — dev scripts.

## Architecture

The framework is built on Julia's **multiple dispatch over a deep abstract type
hierarchy**. A simulation is defined by composing four orthogonal axes —
**Equations × Mesh × Solver × Semidiscretization** — tied together by a time
integrator. Understanding how these combine is the key to the codebase.

- **Type-hierarchy roots** are defined first in `src/basic_types.jl`
  (`AbstractEquations`, `AbstractMesh`, `AbstractSemidiscretization`, …) so they
  are dispatchable throughout the rest of the load pipeline. The include order in
  `src/Trixi.jl` is deliberate: `basic_types` → `equations` → `meshes` → `solvers`
  → parabolic equations → `semidiscretization` (several variants) →
  `time_integration` → `callbacks_step`/`callbacks_stage`. Respect it when adding
  cross-cutting types (parabolic equations load after solvers because they depend
  on parabolic solver types).
- **Equations** (`src/equations/`): each model (e.g. `CompressibleEulerEquations2D`)
  is a concrete type parameterized by dimension and real type, defined mainly by
  *pointwise* functions dispatched on it — `flux`, numerical fluxes
  (`flux_lax_friedrichs`, `flux_ranocha`, …), `initial_condition_*`,
  `boundary_condition_*`, `source_terms_*`, variable conversions (`cons2prim`,
  `cons2entropy`, …).
- **Meshes** (`src/meshes/`): `TreeMesh` (Cartesian, the most basic/most-tested),
  `StructuredMesh`, `UnstructuredMesh2D`, `P4estMesh` (AMR), `T8codeMesh`,
  `DGMultiMesh`. Shared logic is written for `TreeMesh` and specialized only as
  needed for the others.
- **Solvers** (`src/solvers/`): organized as a **mesh × solver matrix**. Base is
  `DGSEM`; directories follow `dgsem_<mesh>` (`dgsem_tree`, `dgsem_p4est`,
  `dgsem_structured`, `dgsem_unstructured`, `dgsem_t8code`). **Shared functionality
  lives in the most basic mesh/solver directory** (so most algorithms are in
  `dgsem_tree/`, and `dgsem_p4est/` only adds what differs). Other families:
  `dgmulti`, `fdsbp_*`, `blockfv`. A solver is further composed from interchangeable
  `VolumeIntegral*` / `SurfaceIntegral*` / `Indicator*` / `Mortar` pieces; the
  shared `rhs!` "recipe" sequences `prolong2interfaces!`, surface/volume integrals,
  etc., and is defined once per dimension and reused across meshes.
- **Semidiscretization** (`src/semidiscretization/`): couples
  mesh + equations + solver into a semi-discrete ODE. `SemidiscretizationHyperbolic`
  is the workhorse; variants cover parabolic, coupled (`…EulerGravity`,
  `…EulerAcoustics`, `…Coupled`), and split systems. `semidiscretize(semi, tspan)`
  returns a SciML `ODEProblem`.
- **Time integration & callbacks**: integrates with OrdinaryDiffEq.jl/SciMLBase
  (`solve(ode, alg; ...)`) plus custom low-storage integrators in
  `src/time_integration/`. **`callbacks_step/`** runs between steps
  (`AnalysisCallback`, `AMRCallback`, `SaveSolutionCallback`, `StepsizeCallback`, …);
  **`callbacks_stage/`** runs within RK stages (limiters, bounds checks).
- **Elixirs** (`examples/`) are the primary user-facing entry point — scripts that
  assemble the above into a runnable simulation (`trixi_include(...)`). The
  `examples/` and `test/` trees both mirror the solver matrix
  (`tree_2d_dgsem`, `p4est_3d_dgsem`, …), and every elixir must be exercised by at
  least one test.

## Code conventions (summary of `docs/src/styleguide.md` + `conventions.md`)

- `CamelCase` for modules/types/structs; lowercase `snake_case` for
  functions/variables. Mutating functions end in `!`.
- 4-space indentation (never tabs). Hard max line length **92**.
- Executable code is ASCII-only; Unicode is fine in docstrings/comments.
- Argument ordering follows Julia Base: the main mutated argument first (e.g. `du`,
  or `cache` when only the cache is mutated), then `mesh, equations, solver, cache`.
  Dispatch-narrowing args go before the general arg (e.g.
  `have_nonconservative_terms(equations), equations`).
- Prefer `for i in ...` over `for i = ...`. Group multiline expressions with
  explicit parentheses rather than relying on implicit continuation.
- Name general-classification-first: `flux_central`, not `central_flux`.
- Naming: suffix `_` (`name_`) for locals that would shadow; prefix `_` (`_name`)
  for fragile/internal, undocumented API.
- Standard elixir keywords (so `trixi_include`/`convergence_test` work):
  `polydeg`, `surface_flux`, `volume_flux`, and `initial_refinement_level` or
  `cells_per_dimension` for resolution.
- Directions: `orientation` `1,2,3 => x,y,z`; `direction` `1..6 => -x,+x,-y,+y,-z,+z`.

## Type stability and numeric types

Trixi.jl supports generic real types (notably `Float32` and `Float64`, plus AD
types) and aims for full type stability. Determine the real type from the relevant
argument (`RealT = eltype(u)`, or `eltype(x)` for coordinates) and `convert`
non-exact constants (`pi`, `1/3`, …) to it. Exact dyadic values (e.g. `0.25`,
`0.5`) may be written as `Float32` literals (`0.25f0`). With a `StepsizeCallback`,
pass an integer dummy `dt` (`dt = 1`, not `1.0`). See `docs/src/conventions.md`
("Numeric types and type stability") for full guidance. Use StaticArrays
(`SVector`/`SMatrix`, `MVector`/`MMatrix`) for small fixed-size arrays.

## Contribution norms

- Branch names follow `<developer-initials>/<short-name>` (e.g. `msl/fix-amr`).
  Develop in a branch and open a PR early (see `docs/src/github-git.md`).
- One focused goal per PR; aim for ≤ 500 changed lines.
- New/changed code must be covered by tests; coverage must not decrease
  (hard floor 97%). Individual tests should run in < 10 s after compilation.
- Document new public functions/types with docstrings; reference relevant
  publications. Note significant changes in `NEWS.md` with the PR number.
- If a PR is created based on major contributions from the LLM/AI tool, disclose LLM/AI assistance in the PR (per `CONTRIBUTING.md`).
- Full checklist: `.github/review-checklist.md`. Conventions and style guide:
  `docs/src/conventions.md`, `docs/src/styleguide.md`.
