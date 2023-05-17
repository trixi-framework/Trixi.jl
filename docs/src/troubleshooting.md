# Troubleshooting and FAQ

In general, Trixi.jl works best with the newest Julia release and up-to-date
dependencies. If something does not work as expected, try updating your installed
Julia packages via the package manager, e.g., by running
```julia
julia> import Pkg; Pkg.update()
```
If you do not use the latest stable release of Julia from the
[official website](https://julialang.org/downloads/#current_stable_release),
consider updating your Julia installation.


## [Installing Trixi.jl as a package only provides an older release](@id old-release)
Trixi.jl requires fairly recent versions of several of its dependencies, which
sometimes causes issues when other installed packages have conflicting version
requirements.  In this case, Julia's package manager `Pkg` will try to handle
this gracefully by going back in history until it finds a Trixi.jl release whose
version requirements can be met, resulting in an older - and usually outdated -
version of Trixi.jl being installed.

The following example illustrates this issue:
* The current Trixi.jl release `v0.3.6` requires package `Foo` with a *minimum* version of `v0.2`.
* An older Trixi.jl release `v0.2.1` requires package `Foo` only with a *minimum*
  version of `v0.1`.
* A user has already installed package `Bar`, which itself requires `Foo` with a
  *maximum* version of `v0.1`.
In this case, installing Trixi.jl via `Pkg` will result in version `v0.2.1` to be
installed instead of the current release `v0.3.6`. That is, a specific release
of Trixi.jl may not be installable if it has a dependency with a higher minimum
version that at the same time is restricted to a lower maximum version by
another installed package.

You can check whether an outdated version of Trixi.jl is installed by executing
```julia
julia> import Pkg; Pkg.update("Trixi"); Pkg.status("Trixi")
```
in the REPL and comparing the reported Trixi.jl version with the version of the
[latest release](https://github.com/trixi-framework/Trixi.jl/releases/latest).
If the versions differ, you can confirm that it is due to a version conflict by
forcing `Pkg` to install the latest Trixi.jl release, where `version` is the
current release:
```julia
julia> Pkg.add(name="Trixi", version="0.3.6")
```
In case of a conflict, the command above will produce an error that informs you
about the offending packages, similar to the following:
```
   Updating registry at `~/.julia/registries/General`
  Resolving package versions...
ERROR: Unsatisfiable requirements detected for package DataStructures [864edb3b]:
 DataStructures [864edb3b] log:
 ├─possible versions are: [0.9.0, 0.10.0, 0.11.0-0.11.1, 0.12.0, 0.13.0, 0.14.0-0.14.1, 0.15.0, 0.16.1, 0.17.0-0.17.20, 0.18.0-0.18.8] or uninstalled
 ├─restricted by compatibility requirements with DiffEqCallbacks [459566f4] to versions: 0.18.0-0.18.8
 │ └─DiffEqCallbacks [459566f4] log:
 │   ├─possible versions are: [2.0.0, 2.1.0, 2.2.0, 2.3.0, 2.4.0, 2.5.0-2.5.2, 2.6.0, 2.7.0, 2.8.0, 2.9.0, 2.10.0, 2.11.0, 2.12.0-2.12.1, 2.13.0-2.13.5, 2.14.0-2.14.1, 2.15.0] or uninstalled
 │   ├─restricted by compatibility requirements with Trixi [a7f1ee26] to versions: [2.14.0-2.14.1, 2.15.0]
 │   │ └─Trixi [a7f1ee26] log:
 │   │   ├─possible versions are: [0.1.0-0.1.2, 0.2.0-0.2.6, 0.3.0-0.3.6] or uninstalled
 │   │   └─restricted to versions 0.3.6 by an explicit requirement, leaving only versions 0.3.6
 │   └─restricted by compatibility requirements with StaticArrays [90137ffa] to versions: 2.15.0 or uninstalled, leaving only versions: 2.15.0
 │     └─StaticArrays [90137ffa] log:
 │       ├─possible versions are: [0.8.0-0.8.3, 0.9.0-0.9.2, 0.10.0, 0.10.2-0.10.3, 0.11.0-0.11.1, 0.12.0-0.12.5, 1.0.0-1.0.1] or uninstalled
 │       └─restricted by compatibility requirements with Trixi [a7f1ee26] to versions: 1.0.0-1.0.1
 │         └─Trixi [a7f1ee26] log: see above
 └─restricted by compatibility requirements with JLD2 [033835bb] to versions: [0.9.0, 0.10.0, 0.11.0-0.11.1, 0.12.0, 0.13.0, 0.14.0-0.14.1, 0.15.0, 0.16.1, 0.17.0-0.17.20] — no versions left
   └─JLD2 [033835bb] log:
     ├─possible versions are: [0.1.0-0.1.14, 0.2.0-0.2.4, 0.3.0-0.3.1] or uninstalled
     └─restricted by compatibility requirements with BinaryBuilder [12aac903] to versions: 0.1.0-0.1.14
       └─BinaryBuilder [12aac903] log:
         ├─possible versions are: [0.1.0-0.1.2, 0.1.4, 0.2.0-0.2.6] or uninstalled
         └─restricted to versions * by an explicit requirement, leaving only versions [0.1.0-0.1.2, 0.1.4, 0.2.0-0.2.6]
```
From the error message, we can see that ultimately `BinaryBuilder` is the
problem here: It restricts the package `DataStructures` to version `v0.17` (via
its dependency `JLD2`), while Trixi.jl requires at least `v0.18` (via its
dependency `DiffEqCallbacks`).
Following the
[official `Pkg` documentation](https://julialang.github.io/Pkg.jl/v1/managing-packages/#conflicts),
there are a number of things you can try to fix such errors:
* Try updating all packages with `julia -e 'using Pkg; Pkg.update()'`. A newer
  version of the problematic package may exist that has updated version
  requirements.
* Remove the offending package. Running
  ```julia
  julia> import Pkg; Pkg.rm("BinaryBuilder"); Pkg.update(); Pkg.status()
  ```
  in the REPL will remove `BinaryBuilder` and (hopefully) update Trixi.jl to the latest version.
* Report the versioning issue to [us](https://github.com/trixi-framework/Trixi.jl/issues/new)
  and/or the development repository of the conflicting package.  Maybe it is
  possible to lift the version restrictions such that both packages can live
  side by side.
* Instead of installing Trixi.jl and conflicting packages in the same (default) environment,
  consider creating new environments/projects and install only packages required for
  the specific tasks, as explained in the
  [official `Pkg` documentation](https://julialang.github.io/Pkg.jl/v1/environments/).
  For example, if you use Trixi.jl for a research project (Bachelor/Master thesis or a paper),
  you should create a new Julia project/environment for that research and add Trixi.jl as a
  dependency. If you track all your code and the `Project.toml`, `Manifest.toml` files
  (generated by `Pkg`) in a version control system such as `git`, you can make your research
  easily reproducible (if you also record the version of Julia you are using and leave some
  comments for others who do not know what you are trying to do, including your future self 😉).


## [There are many questions marks and weird symbols in the output of Trixi.jl](@id font-issues)

This probably means that the default font used by your operating system does not support enough
Unicode symbols. Try installing a modern font with decent unicode support, e.g.
[JuliaMono](https://github.com/cormullion/juliamono). Detailed
[installation instructions](https://cormullion.github.io/pages/2020-07-26-JuliaMono/#installation)
are available there.

This problems affects users of Mac OS particularly often. At the time of writing,
installing JuliaMono is as simple as
```
$ brew tap homebrew/cask-fonts
$ brew install --cask font-juliamono
```


## There are no timing results of the initial mesh creation

By default, the [`SummaryCallback`](@ref) resets the timer used internally by Trixi.jl when it is
initialized (when `solve` is called). If this step needs to be timed, e.g. to debug performance
problems, explicit timings can be used as follows.

```julia
using Trixi

begin
  Trixi.reset_timer!(Trixi.timer())

  equations = LinearScalarAdvectionEquation2D(0.2, -0.7)
  mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), n_cells_max=10^5, initial_refinement_level=5)
  solver = DGSEM(3)
  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

  Trixi.print_timer(Trixi.timer())
end
```


## MPI ranks are assigned zero cells in [`P4estMesh`](@ref) even though there are enough cells

The [`P4estMesh`](@ref) allows one to coarsen the mesh by default. When Trixi.jl is parallelized with multiple MPI
ranks, this has the consequence that sibling cells (i.e., child cells with the same parent cell)
are kept on the same MPI rank to be able to coarsen them easily. This might cause an unbalanced
distribution of cells on different ranks. For 2D meshes, this also means that *initially* each rank will
at least own 4 cells, and for 3D meshes, *initially* each rank will at least own 8 cells.
See [issue #1329](https://github.com/trixi-framework/Trixi.jl/issues/1329).



## Installing and updating everything takes a lot of time

Julia compiles code to get good (C/Fortran-like) performance. At the same time,
Julia provides a dynamic environment and usually compiles code just before using
it. Over time, Julia has improved its caching infrastructure, allowing to store
and reuse more results from (pre-)compilation. This often results in an
increased time to install/update packages, in particular when updating
to Julia v1.8 or v1.9 from older versions.

Some packages used together with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
provide options to configure the amount of precompilation. For example,
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) precompiles
many ODE solvers for a good runtime experience of average users. Currently,
[Trixi.jl](https://github.com/trixi-framework/Trixi.jl) does not use all of
the available solvers. Thus, you can save some time at every update by setting
their [precompilation options](https://docs.sciml.ai/DiffEqDocs/stable/features/low_dep/).

At the time of writing, this could look as follows. First, you need to activate
the environment where you have installed
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl). Then, you need
to execute the following Julia code.

```julia
using Preferences, UUIDs
let uuid = UUID("1dea7af3-3e70-54e6-95c3-0bf5283fa5ed")
  set_preferences!(uuid, "PrecompileAutoSpecialize" => false)
  set_preferences!(uuid, "PrecompileAutoSwitch" => false)
  set_preferences!(uuid, "PrecompileDefaultSpecialize" => true)
  set_preferences!(uuid, "PrecompileFunctionWrapperSpecialize" => false)
  set_preferences!(uuid, "PrecompileLowStorage" => true)
  set_preferences!(uuid, "PrecompileNoSpecialize" => false)
  set_preferences!(uuid, "PrecompileNonStiff" => true)
  set_preferences!(uuid, "PrecompileStiff" => false)
end
```

This disables precompilation of all implicit methods. This should usually not affect
the runtime latency with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
since most setups use explicit time integration methods.
