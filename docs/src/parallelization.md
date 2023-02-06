# Parallelization

## Shared-memory parallelization with threads
Many compute-intensive loops in Trixi.jl are parallelized using the
[multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/)
support provided by Julia. You can recognize those loops by the
`@threaded` macro prefixed to them, e.g.,
```julia
@threaded for element in eachelement(dg, cache)
  ...
end
```
This will statically assign an equal iteration count to each available thread.

To use multi-threading, you need to tell Julia at startup how many threads you
want to use by either setting the environment variable `JULIA_NUM_THREADS` or by
providing the `-t/--threads` command line argument. For example, to start Julia
with four threads, start Julia with
```bash
julia --threads=4
```
If both the environment variable and the command line argument are specified at
the same time, the latter takes precedence.

If you use time integration methods from 
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
and want to use multiple threads therein, you need to set the keyword argument
`thread=OrdinaryDiffEq.True()` of the algorithms, as described in the
[section on time integration methods](@ref time-integration).

!!! warning
    Not everything is parallelized yet and there are likely opportunities to
    improve scalability. Multi-threading isn't considered part of the public
    API of Trixi yet.


## Distributed computing with MPI
In addition to the shared memory parallelization with multi-threading, Trixi.jl
supports distributed parallelism via
[MPI.jl](https://github.com/JuliaParallel/MPI.jl), which leverages the Message
Passing Interface (MPI). MPI.jl comes with its own MPI library binaries such
that there is no need to install MPI yourself. However, it is also possible to
instead use an existing MPI installation, which is recommended if you are
running MPI programs on a cluster or supercomputer
([see the MPI.jl docs](https://juliaparallel.github.io/MPI.jl/stable/configuration/)
to find out how to select the employed MPI library). Additional notes on how to use
a system-provided MPI installation with Trixi.jl can be found in the following subsection.

!!! warning "Work in progress"
    MPI-based parallelization is work in progress and not finished yet. Nothing
    related to MPI is part of the official API of Trixi yet.


### [Using a system-provided MPI installation](@id parallel_system_MPI)

When using Trixi.jl with a system-provided MPI backend the underlying [`p4est`](https://github.com/cburstedde/p4est)
library needs to be compiled with the same MPI installation. Therefore, you also need to use
a system-provided `p4est` installation (for notes on how to install `p4est` see e.g.
[here](https://github.com/cburstedde/p4est/blob/master/README), use the configure option
`--enable-mpi`). In addition, [P4est.jl](https://github.com/trixi-framework/P4est.jl) needs to
be configured to use the custom `p4est` installation. Follow the steps described
[here](https://github.com/trixi-framework/P4est.jl/blob/main/README.md) for the configuration.
In total, in your active Julia project you should have a LocalPreferences.toml file with sections
`[MPIPreferences]` and `[P4est]` as well as an entry `MPIPreferences` in your Project.toml to
use a custom MPI installation.


### [Usage](@id parallel_usage)

To start Trixi in parallel with MPI, there are three options:

1. **Run from the REPL with `mpiexec()`:** You can start a parallel execution directly from the
   REPL by executing
   ```julia
   julia> using MPI

   julia> mpiexec() do cmd
            run(`$cmd -n 3 $(Base.julia_cmd()) --threads=1 --project=@. -e 'using Trixi; trixi_include(default_example())'`)
          end
   ```
   The parameter `-n 3` specifies that Trixi should run with three processes (or
   *ranks* in MPI parlance) and should be adapted to your available
   computing resources and problem size. The `$(Base.julia_cmd())` argument
   ensures that Julia is executed in parallel with the same optimization level
   etc. as you used for the REPL; if this is unnecessary or undesired, you can
   also just use `julia`.  Further, if you are not running Trixi from a local
   clone but have installed it as a package, you need to omit the `--project=@.`.
2. **Run from the command line with `mpiexecjl`:** Alternatively, you can
   use the `mpiexecjl` script provided by MPI.jl, which allows you to start
   Trixi in parallel directly from the command line. As a preparation, you need to
   install the script *once* by running
   ```julia
   julia> using MPI

   julia> MPI.install_mpiexecjl(destdir="/somewhere/in/your/PATH")
   ```
   Then, to execute Trixi in parallel, execute the following command from your
   command line:
   ```bash
   mpiexecjl -n 3 julia --threads=1 --project=@. -e 'using Trixi; trixi_include(default_example())'
   ```
3. **Run interactively with `tmpi` (Linux/MacOS only):** If you are on a
   Linux/macOS system, you have a third option which lets you run Julia in
   parallel interactively from the REPL. This comes in handy especially during
   development, as in contrast to the first two options, it allows to reuse the
   compilation cache and thus facilitates much faster startup times after the
   first execution. It requires [tmux](https://github.com/tmux/tmux) and the
   [OpenMPI](https://www.open-mpi.org) library to be installed before, both of
   which are usually available through a package manager. Once you have
   installed both tools, you need to configure MPI.jl to use the OpenMPI for
   your system, which is explained
   [here](https://juliaparallel.org/MPI.jl/stable/configuration/#Using-a-system-provided-MPI-backend).
   Then, you can download and install the
   [tmpi](https://github.com/Azrael3000/tmpi)
   script by executing
   ```bash
   curl https://raw.githubusercontent.com/Azrael3000/tmpi/master/tmpi -o /somewhere/in/your/PATH/tmpi
   ```
   Finally, you can start and control multiple Julia REPLs simultaneously by
   running
   ```bash
   tmpi 3 julia --threads=1 --project=@.
   ```
   This will start Julia inside `tmux` three times and multiplexes all commands
   you enter in one REPL to all other REPLs (try for yourself to understand what
   it means). If you have no prior experience with `tmux`, handling the REPL
   this way feels slightly weird in the beginning. However, there is a lot of
   documentation for `tmux`
   [available](https://github.com/tmux/tmux/wiki/Getting-Started) and once you
   get the hang of it, developing Trixi in parallel becomes much smoother this
   way. Some helpful commands are the following. To close a single pane you can press `Ctrl+b`
   and then `x` followed by `y` to confirm. To quit the whole session you press `Ctrl+b` followed
   by `:kill-session`. Often you would like to scroll up. You can do that by pressing `Ctrl+b` and then `[`,
   which allows you to use the arrow keys to scroll up and down. To leave the scroll mode you press `q`.
   Switching between panes can be done by `Ctrl+b` followed by `o`.
   As of March 2022, newer versions of tmpi also support mpich, which is the default
   backend of MPI.jl (via MPICH_Jll.jl). To use this setup, you need to install
   `mpiexecjl` as described in the 
   [documentation of MPI.jl](https://juliaparallel.org/MPI.jl/v0.20/usage/#Julia-wrapper-for-mpiexec)
   and make it available as `mpirun`, e.g., via a symlink of the form
   ```bash
   ln -s ~/.julia/bin/mpiexecjl /somewhere/in/your/path/mpirun
   ```
   (assuming default installations).

!!! note "Hybrid parallelism"
    It is possible to combine MPI with shared memory parallelism via threads by starting
    Julia with more than one thread, e.g. by passing the command line argument
    `julia --threads=2` instead of `julia --threads=1` used in the examples above.
    In that case, you should make sure that your system supports the number of processes/threads
    that you try to start.


### [Performance](@id parallel_performance)
For information on how to evaluate the parallel performance of Trixi, please
have a look at the [Performance metrics of the `AnalysisCallback`](@ref)
section, specifically at the descriptions of the performance index (PID).
