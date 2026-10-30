# The `run_agents` Julia projects: details

This file complements the section "The `run_agents` Julia projects" in
[`AGENTS.md`](../../AGENTS.md). Read it when setting up or updating the
projects fails, when loading Trixi.jl fails with a missing dependency, when you
need additional packages, or when comparing results with CI.

## Layout

- `run_agents/` develops the local version of Trixi.jl (`Pkg.develop`) and
  contains all dependencies of the tests (copied from `test/Project.toml`,
  including compat bounds and preferences).
- `run_agents/kernelabstractions/` is a copy of `run_agents/` (same project and
  manifest) whose `LocalPreferences.toml` sets the threading backend of
  Trixi.jl to KernelAbstractions.jl, as in the CI jobs `kernelabstractions` and
  `kernelabstractions_mpi`. The setup script overwrites changes made directly
  in this directory whenever `run_agents/` differs.
- `run_agents/formatter/` contains JuliaFormatter.jl v1.0.60, which cannot be
  installed in `run_agents/` since its dependencies are incompatible with those
  of Trixi.jl.

The script `utils/setup_run_agents.jl` works from any directory and is
idempotent: if `run_agents/` is up to date with `test/Project.toml` and still
points to this checkout, it reuses the environment; otherwise, it recreates the
project. It only rewrites `run_agents/kernelabstractions/` if it differs from
`run_agents/`, and it only (re-)creates `run_agents/formatter/` if
JuliaFormatter.jl v1.0.60 is not pinned there. The first setup may take several
minutes; afterward, it takes only a few seconds unless packages need to be
precompiled again.

## Missing dependencies of Trixi.jl

The setup script does not check the dependencies of Trixi.jl itself (in the
root `Project.toml`). If they changed (e.g., after you added a dependency to
Trixi.jl or pulled such a change from `main`), loading Trixi.jl fails with an
error such as `ArgumentError: Package Trixi does not have X in its dependencies`.
Only in this case, run
`julia --project=run_agents -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`
and then the setup script (which copies the new manifest to
`run_agents/kernelabstractions/`). Do not run `Pkg.resolve()` routinely, since
it takes some time.

## Additional packages

If you need additional packages for your own experiments (e.g.,
BenchmarkTools), add them with
`julia --project=run_agents -e 'using Pkg; Pkg.add("PackageName")'`. Note that
they are removed when the project is re-created after a change of
`test/Project.toml`; never add them to `test/Project.toml` or `Project.toml`
unless this is part of the actual task. Temporary scripts can be stored in
`run_agents/` as well.

## Package versions and CI

`run_agents/Manifest.toml` pins resolved package versions, and the setup script
does not upgrade them. CI uses the latest compatible versions, so the local
environment can fall behind. When investigating CI failures or numerical
differences, compare the package versions in the CI logs with
`julia --project=run_agents -e 'using Pkg; Pkg.status()'` before attributing a
change to Trixi.jl. To move to the latest compatible versions, run
`julia --project=run_agents -e 'using Pkg; Pkg.Registry.update(); Pkg.update()'`.
Use a separate ignored or temporary environment to test other dependency
versions so that the usual `run_agents/` setup remains reproducible.
