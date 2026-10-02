# AGENTS.md

Canonical instructions for AI coding agents (Claude Code, Codex, Copilot,
Cursor, Gemini CLI, ...) working on Trixi.jl. Most tools read this file natively;
Claude Code reads it via the `@AGENTS.md` import in `CLAUDE.md`. Keep this file
the single source of truth; put only tool-specific notes into tool-specific
files. The authoritative developer documentation lives in `docs/src/` (in
particular `conventions.md`, `styleguide.md`, `testing.md`, and
`development.md`); this file summarizes what agents need most.

When editing this file, only add instructions and commands that have actually
been tested in this repository. Instructions that do not work are worse than no
instructions.

Trixi.jl is a Julia package for adaptive high-order numerical simulations of
conservation laws (hyperbolic PDEs and related parabolic and multi-physics
problems), mainly with discontinuous Galerkin spectral element methods (DGSEM).
Correctness of the numerics matters more than anything else: many bugs are
silent (e.g., a wrong index or sign that only reduces accuracy), so be careful
and verify mathematical changes.

Before editing, inspect `git status` and preserve all existing work. Do not
discard, reset, or stash unrelated changes while testing a fix.

## Repository layout

- `src/`: package code (see [Architecture](#architecture)).
- `examples/`: elixirs, i.e., scripts setting up and running a simulation,
  grouped by mesh, dimension, and solver (e.g., `tree_2d_dgsem`,
  `p4est_3d_dgsem`).
- `test/`: test suite (entry point `test/runtests.jl`, test items in
  `test/test_*.jl`).
- `ext/`: package extensions for weak dependencies (e.g., Makie, Plots,
  CUDA, AMDGPU).
- `docs/`: documentation source; `utils/`: developer scripts (e.g., formatting).
- `run/` (if present, ignored by git): the human developer's own Julia project
  as suggested in `README.md`. Do not modify it; agents use `run_agents/`
  instead (see below).

## Architecture

Trixi.jl is built on multiple dispatch over an abstract type hierarchy. A
simulation combines **equations × mesh × solver** into a **semidiscretization**,
which is integrated in time.

- The abstract root types (`AbstractEquations`, `AbstractMesh`,
  `AbstractSemidiscretization`, ...) are defined in `src/basic_types.jl`. The
  include order in `src/Trixi.jl` is deliberate (`basic_types` → `equations` →
  `meshes` → `solvers` → parabolic equations → `semidiscretization` →
  `time_integration` → `callbacks_step`/`callbacks_stage`); respect it when
  adding cross-cutting types.
- **Equations** (`src/equations/`), e.g., `CompressibleEulerEquations2D`, are
  concrete types defined mainly by *pointwise* functions dispatched on them:
  `flux`, numerical fluxes (`flux_lax_friedrichs`, `flux_ranocha`, ...), wave
  speed estimates, `initial_condition_*`, `boundary_condition_*`,
  `source_terms_*`, and variable conversions (`cons2prim`, `cons2entropy`, ...).
- **Meshes** (`src/meshes/`): `TreeMesh` (Cartesian, the most basic and most
  tested), `StructuredMesh`, `UnstructuredMesh2D`, `P4estMesh`, `T8codeMesh`,
  and `DGMultiMesh`.
- **Solvers** (`src/solvers/`) form a mesh × solver matrix: DGSEM in
  `dgsem_tree`, `dgsem_structured`, `dgsem_unstructured`, `dgsem_p4est`,
  `dgsem_t8code` (plus common parts in `dgsem`), and other families in
  `dgmulti`, `fdsbp_*`, and `blockfv`. **Shared functionality lives in the
  directory of the most basic mesh/solver type** (mostly `dgsem_tree/`); the
  other directories only add what differs. A DGSEM solver is composed of
  interchangeable `VolumeIntegral*`, `SurfaceIntegral*`, `Indicator*`, and
  mortar components. The right-hand side `rhs_hyperbolic!` (and
  `rhs_parabolic!`) sequences steps such as `prolong2interfaces!`, interface,
  boundary, and mortar fluxes, and volume and surface integrals.
- **Semidiscretizations** (`src/semidiscretization/`) couple mesh, equations,
  and solver. `SemidiscretizationHyperbolic` is the workhorse; variants cover
  hyperbolic-parabolic, coupled, split, Euler-gravity, and Euler-acoustics
  systems. `semidiscretize(semi, tspan)` returns an `ODEProblem`.
- **Time integration and callbacks**: OrdinaryDiffEq.jl/SciMLBase
  (`solve(ode, alg; ...)`) plus custom integrators in `src/time_integration/`
  (low-storage, SSP, paired explicit RK, relaxation methods).
  `src/callbacks_step/` contains callbacks between time steps (e.g.,
  `AnalysisCallback`, `AMRCallback`, `SaveSolutionCallback`,
  `StepsizeCallback`); `src/callbacks_stage/` contains stage limiters and
  bounds checks applied within Runge-Kutta stages.
- **Elixirs** (`examples/`) are the main user-facing entry point and are run via
  `trixi_include`. Every elixir must be used in at least one test.

## Testing: never run the full test suite

**Never run the complete test suite of Trixi.jl.** It takes many hours and is
run in CI (GitHub Actions) for every pull request. In particular, do **not** run

- `Pkg.test("Trixi")` or `julia test/runtests.jl` without a restrictive setup
  (without `TRIXI_TEST`, they run the `threaded` suite by default),
- `TRIXI_TEST=all`, or entire CI partitions such as `TRIXI_TEST=tree_part1`,
- whole test files such as `include("test/test_tree_2d_euler.jl")`,
- the special suites `mpi`, `threaded`, `threaded_legacy`, `downgrade`,
  `kernelabstractions`, `CUDA`, or `AMDGPU`.

Instead, run only the individual test items related to the code you changed.
Tests are isolated `@testitem`s (TestItems.jl) run with TestItemRunner.jl. Test
files are named like `test_<mesh>_<dim>_<equation>.jl` (e.g.,
`test_tree_2d_euler.jl`) plus files such as `test_unit.jl`. Each test item
lists its setup snippets via `setup = [...]` (e.g., `Setup`, defined in
`test/runtests.jl`, and file-specific snippets such as `TreeMesh2DMHD`, defined
at the top of the corresponding test file) and its CI job via `tags = [...]`;
CI runs one job per value of the `TRIXI_TEST` variable, which selects the test
items with the corresponding tag (see `test/runtests.jl` and
`.github/workflows/ci.yml`).

### The `run_agents` Julia project

Use a local Julia project in the directory `run_agents/` at the root of this
repository for running tests, elixirs, and scratch scripts. It develops the
local version of Trixi.jl (`Pkg.develop`) and contains all dependencies of the
tests (copied from `test/Project.toml`, including compat bounds and
preferences). Recycle it from one run (and session) to the next instead of
creating new environments. The directory is ignored by git (`run*/` in
`.gitignore`) and must never be committed.

The formatter JuliaFormatter.jl v1.0.60 (see [Code style](#code-style)) cannot
be installed in `run_agents/` since its dependencies are incompatible with those
of Trixi.jl. Thus, it lives in the separate project `run_agents/formatter/`,
where it is pinned to this version.

Create or update both projects by running the following command from the root
of the repository **before running tests or the formatter**. It is idempotent:
if `run_agents/` is up to date with `test/Project.toml` and still points to this
checkout, it reuses the environment; otherwise, it recreates the project.
Similarly, it only (re-)creates `run_agents/formatter/` if JuliaFormatter.jl
v1.0.60 is not pinned there. The first setup may take several minutes.

```bash
julia -e '
using Pkg, TOML
mkpath("run_agents")
test_project = TOML.parsefile(joinpath("test", "Project.toml"))
project_file = joinpath("run_agents", "Project.toml")
manifest_file = joinpath("run_agents", "Manifest.toml")
project = isfile(project_file) ? TOML.parsefile(project_file) : Dict{String, Any}()
manifest = isfile(manifest_file) ? TOML.parsefile(manifest_file) : Dict{String, Any}()
deps = get(project, "deps", Dict{String, Any}())
trixi_entries = get(get(manifest, "deps", Dict{String, Any}()), "Trixi", [])
test_deps_present = issubset(keys(test_project["deps"]), keys(deps))
sections_match = all(get(project, section, nothing) ==
                     get(test_project, section, nothing)
                     for section in ("compat", "extras", "preferences", "targets"))
local_trixi = haskey(deps, "Trixi") &&
              any(get(entry, "path", nothing) == ".." for entry in trixi_entries)
up_to_date = test_deps_present && sections_match && local_trixi
Pkg.activate("run_agents")
if !up_to_date
    cp(joinpath("test", "Project.toml"), project_file; force = true)
    Pkg.develop(path = ".")
end
Pkg.instantiate()
formatter_dir = joinpath("run_agents", "formatter")
formatter_manifest_file = joinpath(formatter_dir, "Manifest.toml")
formatter_manifest = isfile(formatter_manifest_file) ?
                     TOML.parsefile(formatter_manifest_file) : Dict{String, Any}()
formatter_entries = get(get(formatter_manifest, "deps", Dict{String, Any}()),
                        "JuliaFormatter", [])
formatter_pinned = any(get(entry, "version", nothing) == "1.0.60" &&
                       get(entry, "pinned", false) for entry in formatter_entries)
formatter_pinned || rm(formatter_dir; force = true, recursive = true)
Pkg.activate(formatter_dir)
if !formatter_pinned
    Pkg.add(name = "JuliaFormatter", version = "1.0.60")
    Pkg.pin("JuliaFormatter")
end
Pkg.instantiate()'
```

The setup command does not check the dependencies of Trixi.jl itself (in the
root `Project.toml`). If they changed (e.g., after you added a dependency to
Trixi.jl or pulled such a change from `main`), loading Trixi.jl fails with an
error such as `ArgumentError: Package Trixi does not have X in its
dependencies`. Only in this case, run
`julia --project=run_agents -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`.
Do not run `Pkg.resolve()` routinely, since it takes some time.

If you need additional packages for your own experiments (e.g., BenchmarkTools),
add them with `julia --project=run_agents -e 'using Pkg; Pkg.add("PackageName")'`.
Note that they are removed when the project is re-created after a change of
`test/Project.toml`; never add them to `test/Project.toml` or `Project.toml`
unless this is part of the actual task. Temporary scripts can be stored in
`run_agents/` as well.

`run_agents/Manifest.toml` pins resolved package versions, and the setup command
above does not upgrade them. CI uses the latest compatible versions, so the
local environment can fall behind. When investigating CI failures or numerical
differences, compare the package versions in the CI logs with
`julia --project=run_agents -e 'using Pkg; Pkg.status()'` before attributing a
change to Trixi.jl. To move to the latest compatible versions, run
`julia --project=run_agents -e 'using Pkg; Pkg.Registry.update(); Pkg.update()'`.
Use a separate ignored or temporary environment to test other dependency
versions so that the usual `run_agents/` setup remains reproducible.

### Running selected test items

From the root of the repository, run, e.g.,

```bash
julia --project=run_agents -e '
using TestItemRunner
TestItemRunner.run_tests(pwd(); filter = ti -> ti.name in (
    "TreeMesh2D Advection: elixir_advection_basic.jl",
    "Unit: Spectral analysis",
))'
```

Useful filters (`ti` has the fields `name`, `filename`, and `tags`):

- exact names: `ti -> ti.name in ("name 1", "name 2")`
- substring: `ti -> occursin("elixir_mhd_alfven_wave", ti.name)`
- one file (use sparingly, some files are large):
  `ti -> endswith(ti.filename, "test_tree_2d_mhdmulti.jl")`

Notes:

- **Check the test summary.** A filter that matches no test item does not fail;
  TestItemRunner just prints a summary without any tests (`None`). Make sure
  that the number of passed tests is plausible, and copy test item names
  exactly from the `@testitem "..."` lines.
- If a persistent Julia session is available, prefer it (see
  [the next section](#optional-persistent-julia-session-via-mcp)). Otherwise,
  batch related test items into one call to avoid repeated startup costs.
- `TestItemRunner.run_tests(pwd(); ...)` works from the repository root.
  `@run_package_tests filter = ...` (as used in `docs/src/testing.md`) only
  works when the current directory is `test/` (e.g., after `cd("test")`); from
  the repository root, `Trixi` is not loaded inside the test items
  (`UndefVarError: Trixi not defined`).
- Do not use `julia --project=.` (the package environment lacks the test
  dependencies) or `julia --project=test` (the test environment lacks
  Trixi.jl), and do not stack environments via `JULIA_LOAD_PATH` (e.g.,
  `"@:test:@stdlib"`); the latter can mix incompatible package versions and fail
  during precompilation. Use the `run_agents` project instead.
- The first run after setting up or updating `run_agents` may take several
  minutes because of precompilation. Later runs are much faster.
- Find the relevant test items by searching `test/` for the elixir name,
  function name, or equation type you changed, e.g.,
  `rg -n "elixir_mhd_rotor" test/` (or `grep -rn`). Remember that elixirs can
  be included by other elixirs and tests via `trixi_include`.
- To quickly try an elixir without the test harness, use the `run_agents`
  project (elixirs need packages such as `OrdinaryDiffEqLowStorageRK`, which are
  test-only dependencies) and `trixi_include`, which can override assignments in
  the elixir, e.g., in `julia --project=run_agents`:
  ```julia
  using Trixi
  trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_extended.jl"),
                tspan = (0.0, 0.1))
  ```
  Only variables that are assigned in the elixir can be overridden (e.g., not
  every elixir defines `tspan`). Output files are written to `out/`, which must
  not be committed.
- When developing new kernels, run Julia with `--check-bounds=yes` so that
  out-of-bounds accesses are caught despite `@inbounds`.

### Optional: persistent Julia session via MCP

Some developers configure the MCP server
[julia-mcp](https://github.com/aplavin/julia-mcp), which provides the tools
`julia_eval`, `julia_restart`, and `julia_list_sessions` (in Claude Code, e.g.,
`mcp__julia__julia_eval`). **Only if these tools are available to you**, use
`julia_eval` to run Julia code such as tests and elixirs: the Julia session
persists between calls, so packages are loaded and compiled only once. If the
tools are not available, use the shell commands described above; everything
else in this file applies unchanged. Run the `run_agents` setup command in the
shell in any case, even if the description of `julia_eval` tells you to never
run Julia via the command line.

- Always pass the **absolute path** of the `run_agents` directory as
  `env_path` (e.g., `/path/to/Trixi.jl/run_agents`). Do not call
  `Pkg.activate` in the code.
- The working directory of the session is `run_agents/`, not the repository
  root. Use absolute paths, e.g.,
  `TestItemRunner.run_tests("/path/to/Trixi.jl"; filter = ...)`.
- The default `timeout` is 60 seconds, and a call exceeding its timeout **kills
  the session**. Pass a generous `timeout` (e.g., `1800`) for the first call
  (`using Trixi, TestItemRunner`) and for the first run of test items or
  elixirs, which include compilation.
- Afterwards, repeat `TestItemRunner.run_tests(...)` calls with different
  filters in the same session. Revise.jl is loaded automatically (if it is
  installed in the global environment), so changes to files in `src/` are
  picked up without restarting; test files are re-read by TestItemRunner.
- Call `julia_restart` with the same `env_path` after changing type
  definitions (e.g., fields of a `struct`), which Revise cannot handle, after
  re-creating or updating `run_agents`, or if the session is in a broken state.
- The session may run with non-default Julia flags chosen by the developer
  (recommended: `--threads=1 --check-bounds=yes --startup-file=no`). Check
  them with `Threads.nthreads()` and `Base.JLOptions().check_bounds` (`1` means
  bounds checking is enabled) if they matter, e.g., for timings or allocation
  tests.
- `trixi_include` writes its output files to `out/` in the working directory,
  i.e., to `run_agents/out/`.
- Run the formatter (see [Code style](#code-style)) in a separate session with
  the absolute path of `run_agents/formatter` as `env_path`, e.g.,
  `using JuliaFormatter; format(["/path/to/Trixi.jl/src/file.jl"])`, since
  JuliaFormatter.jl is not available in the `run_agents` environment. Use
  absolute paths since the working directory is `run_agents/formatter/`.

### Adding and changing tests

- New or changed behavior must be covered by focused tests; the overall test
  coverage must not decrease (hard lower bound 97%). For a bug fix, verify that
  the regression test detects the original behavior when feasible, using a
  separate checkout or temporarily reverting only your own changes. Preserve
  unrelated work in the shared working tree.
- Add tests as `@testitem`s to the matching `test/test_xxx.jl` file (e.g., 3D
  linear advection on the `TreeMesh` goes into `test/test_tree_3d_advection.jl`,
  unit tests into `test/test_unit.jl`). Copy the `setup = [...]` and
  `tags = [...]` of neighboring test items; the tags select the CI job.
- Exercise every new elixir in at least one test item. For a new equation or
  numerical method, run a convergence study and report the observed order.
- Test item names must be unique within a file.
- Each test item should run in **less than 10 seconds** (single thread, after
  compilation). Use short `tspan`s or coarse meshes where possible.
- Regression tests via `@test_trixi_include` compare `l2`/`linf` errors with
  reference values. If a change intentionally modifies results, inspect the
  actual values and relevant numerical behavior first, and check that the change
  is plausible; for example, errors of smooth test cases should not increase
  after a bug fix. Then run the affected test items once, take the actual values
  from the `Evaluated: isapprox(expected, actual; ...)` lines of the failures,
  replace the reference values **only within the affected test items**, rerun
  them, and explain the change in the PR.
- Never loosen tolerances (`atol`, `rtol`) just to make a failing test pass.
  A local, commented tolerance is acceptable only for small differences whose
  cause is understood and outside Trixi.jl (e.g., adaptive time stepping in a
  new dependency version); mention it to the user.
- Unit tests for numerical fluxes and wave speed estimates should use
  *different* left and right states: many copy-paste bugs are invisible for
  `u_ll == u_rr`.

## Code style

- Format all changed Julia files with JuliaFormatter.jl v1.0.60 (configuration
  in `.JuliaFormatter.toml`, SciML style); CI checks the formatting. Other
  versions of JuliaFormatter.jl format differently, so use the pinned version
  in `run_agents/formatter/` (created by the
  [setup command](#the-run_agents-julia-project)), e.g., from the root of the
  repository,
  ```bash
  julia --project=run_agents/formatter -e 'using JuliaFormatter; format(["src/file.jl", "test/test_file.jl"])'
  ```
  or via the MCP server if available (see
  [above](#optional-persistent-julia-session-via-mcp)). `format` returns `true`
  if all files were already formatted; otherwise, it formats them in place and
  returns `false`. Pass only the files you changed. (Human developers often use
  `utils/trixi-format-file.jl`, which installs the formatter in a temporary
  environment on every call.)
- Maximum line length 92, 4 spaces for indentation (never tabs), `snake_case`
  for functions and variables, `CamelCase` for modules and types, trailing `!`
  for mutating functions. Executable code must be ASCII; comments and docstrings
  may use Unicode.
- Argument order: the modified argument first (e.g., `du`, or `cache` if only
  the cache is modified), otherwise `mesh, equations, solver, cache`; dispatch
  helpers before the general argument (e.g.,
  `have_nonconservative_terms(equations), equations`).
- Name related functions with the general category first, e.g.,
  `flux_central` (not `central_flux`), `min_max_speed_davis`.
- Use a suffix `_` (`name_`) for local variables that would otherwise shadow
  existing names and a prefix `_` (`_name`) for fragile internal functions that
  are not part of the documented API.
- Prefer `for i in ...` over `for i = ...`. Group multiline expressions with
  explicit parentheses instead of relying on implicit line continuation.
- Most source files are wrapped in `@muladd begin ... end`; keep this structure.
- Keep code type-stable and generic in the real type (`Float32` and `Float64`
  are supported, as well as AD types): use `0.5f0` for exactly representable
  constants and `convert(RealT, pi)` with `RealT = eltype(u)` (or `eltype(x)`)
  otherwise. Use StaticArrays.jl (`SVector`, `MVector`, ...) for small arrays of
  fixed size.
- Performance-critical code must not allocate; many regression tests check this
  with `@test_allocations`.
- Document new public functions and types with docstrings; for internal code,
  a top-level comment can suffice. Reference publications where appropriate.
  Comments should explain *why*, not only *what*.
- Match the style of the surrounding code, and look for existing helpers before
  adding new ones.
- In elixirs, use the standard variable names `polydeg`, `surface_flux`,
  `volume_flux`, and `initial_refinement_level` or `cells_per_dimension` for the
  resolution, so that `trixi_include` and `convergence_test` can modify them.
  With a `StepsizeCallback`, pass an integer dummy time step `dt = 1` (not
  `1.0`) to `solve`.

## Numerical conventions and common pitfalls

- `orientation`s: `1 => x, 2 => y, 3 => z`; `direction`s:
  `1 => -x, 2 => +x, 3 => -y, 4 => +y, 5 => -z, 6 => +z`.
- A `normal_direction` argument may be non-unit and carry surface scaling,
  especially on curved meshes. Fluxes and wave speeds using it must account
  for its magnitude exactly once. Code such as `FluxRotated` explicitly forms
  a unit `normal_vector` and restores the magnitude afterward. Numerical fluxes
  and wave speed estimates must be homogeneous of degree one in
  `normal_direction`, i.e., `f(u_ll, u_rr, α * n) == α * f(u_ll, u_rr, n)` for
  `α > 0` (e.g., multiply sound speeds by `norm(normal_direction)`). Check
  agreement with the corresponding `orientation` method for axis-aligned unit
  normals.
- Nonconservative surface terms depend on the argument order: the element
  receiving the flux passes its own state first,
  `nonconservative_flux(u_own, u_neighbor, ...)`, including at mortars.
- Arrays visible to the time integrator have the suffix `_ode` (`u_ode`,
  `du_ode`); when passed with `semi`, normally wrap them with
  `wrap_array(u_ode, semi)` before internal use. Wrapped arrays have no suffix.
  Use `wrap_array_native` only where needed for external interfaces, and
  document exceptions such as AMR.
- Compare 1D/2D/3D and x/y/z implementations of the same function when
  changing one of them; copy-paste errors between dimensions and directions are
  the most common source of bugs.

## Pull requests and documentation

- Develop in a branch; branch names usually follow
  `<developer initials>/<short description>` (e.g., `hr/fix_mortar_fluxes`).
  See `docs/src/github-git.md` for the general workflow.
- Keep changes focused on a single goal; PRs should usually change at most about
  500 lines. See `.github/review-checklist.md` for what reviewers look for.
- Document significant user-visible changes, including bug fixes that change
  results, in `NEWS.md` (section of the current lifecycle, e.g., "Added" or
  "Changed", with the PR number once known).
- If a PR is created based on major contributions from an LLM/AI tool, disclose
  the LLM/AI assistance in the PR description (see `CONTRIBUTING.md`).
- Do not commit generated files (e.g., `out*/` directories, `.vtu`/`.h5`
  output, `Manifest.toml` files) or temporary notes.
- Do not commit or push unless asked to do so by the user.
