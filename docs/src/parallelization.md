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
    API of Trixi.jl yet.


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
    related to MPI is part of the official API of Trixi.jl yet.


### [Using a system-provided MPI installation](@id parallel_system_MPI)

When using Trixi.jl with a system-provided MPI backend, the underlying
[`p4est`](https://github.com/cburstedde/p4est), [`t8code`](https://github.com/DLR-AMR/t8code)
and [`HDF5`](https://github.com/HDFGroup/hdf5) libraries need to be compiled with the same MPI
installation. If you want to use `p4est` (via the `P4estMesh`) or `t8code` (via the `T8codeMesh`)
from Trixi.jl, you also need to use system-provided `p4est` or `t8code` installations
(for notes on how to install `p4est` and `t8code` see, e.g., [here](https://github.com/cburstedde/p4est/blob/master/README)
and [here](https://github.com/DLR-AMR/t8code/wiki/Installation), use the configure option
`--enable-mpi`). Otherwise, there will be warnings that no preference is set for P4est.jl and
T8code.jl that can be ignored if you do not use these libraries from Trixi.jl. Note that
`t8code` already comes with a `p4est` installation, so it suffices to install `t8code`.
In order to use system-provided `p4est` and `t8code` installations, [P4est.jl](https://github.com/trixi-framework/P4est.jl)
and [T8code.jl](https://github.com/DLR-AMR/T8code.jl) need to be configured to use the custom
installations. Follow the steps described [here](https://github.com/DLR-AMR/T8code.jl/blob/main/README.md#installation) and
[here](https://github.com/trixi-framework/P4est.jl/blob/main/README.md#installation) for the configuration.
The paths that point to `libp4est.so` (and potentially to `libsc.so`) need to be
the same for P4est.jl and T8code.jl. This could, e.g., be `libp4est.so` that usually can be found
in `lib/` or `local/lib/` in the installation directory of `t8code`.
The preferences for [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) always need to be set, even if you
do not want to use `HDF5` from Trixi.jl, see also [issue #1079 in HDF5.jl](https://github.com/JuliaIO/HDF5.jl/issues/1079).
To set the preferences for HDF5.jl, follow the instructions described
[here](https://trixi-framework.github.io/Trixi.jl/stable/parallelization/#Using-parallel-input-and-output).

In total, in your active Julia project you should have a `LocalPreferences.toml` file with sections
`[MPIPreferences]`, `[T8code]` (only needed if `T8codeMesh` is used), `[P4est]` (only needed if
`P4estMesh` is used), and `[HDF5]` as well as an entry `MPIPreferences` in your
`Project.toml` to use a custom MPI installation. A `LocalPreferences.toml` file 
created as described above might look something like the following:
```toml
[HDF5]
libhdf5 = "/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so"
libhdf5_hl = "/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5_hl.so"

[HDF5_jll]
libhdf5_hl_path = "/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5_hl.so"
libhdf5_path = "/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so"

[MPIPreferences]
__clear__ = ["preloads_env_switch"]
_format = "1.0"
abi = "OpenMPI"
binary = "system"
cclibs = []
libmpi = "/lib/x86_64-linux-gnu/libmpi.so"
mpiexec = "mpiexec"
preloads = []

[P4est]
libp4est = "/home/mschlott/hackathon/libtrixi/t8code/install/lib/libp4est.so"
libsc = "/home/mschlott/hackathon/libtrixi/t8code/install/lib/libsc.so"

[T8code]
libp4est = "/home/mschlott/hackathon/libtrixi/t8code/install/lib/libp4est.so"
libsc = "/home/mschlott/hackathon/libtrixi/t8code/install/lib/libsc.so"
libt8 = "/home/mschlott/hackathon/libtrixi/t8code/install/lib/libt8.so"
```

This file is created with the following sequence of commands:
```julia
julia> using MPIPreferences
julia> MPIPreferences.use_system_binary()
```
Restart the Julia REPL
```julia
julia> using P4est
julia> P4est.set_library_p4est!("/home/mschlott/hackathon/libtrixi/t8code/install/lib/libp4est.so")
julia> P4est.set_library_sc!("/home/mschlott/hackathon/libtrixi/t8code/install/lib/libsc.so")
julia> using T8code
julia> T8code.set_libraries_path!("/home/mschlott/hackathon/libtrixi/t8code/install/lib/")
julia> using HDF5
julia> HDF5.API.set_libraries!("/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so", "/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5_hl.so")
```
After the preferences are set, restart the Julia REPL again.

### [Usage](@id parallel_usage)

To start Trixi.jl in parallel with MPI, there are three options:

1. **Run from the REPL with `mpiexec()`:** You can start a parallel execution directly from the
   REPL by executing
   ```julia
   julia> using MPI

   julia> mpiexec() do cmd
            run(`$cmd -n 3 $(Base.julia_cmd()) --threads=1 --project=@. -e 'using Trixi; trixi_include(default_example())'`)
          end
   ```
   The parameter `-n 3` specifies that Trixi.jl should run with three processes (or
   *ranks* in MPI parlance) and should be adapted to your available
   computing resources and problem size. The `$(Base.julia_cmd())` argument
   ensures that Julia is executed in parallel with the same optimization level
   etc. as you used for the REPL; if this is unnecessary or undesired, you can
   also just use `julia`.  Further, if you are not running Trixi.jl from a local
   clone but have installed it as a package, you need to omit the `--project=@.`.
2. **Run from the command line with `mpiexecjl`:** Alternatively, you can
   use the `mpiexecjl` script provided by MPI.jl, which allows you to start
   Trixi.jl in parallel directly from the command line. As a preparation, you need to
   install the script *once* by running
   ```julia
   julia> using MPI

   julia> MPI.install_mpiexecjl(destdir="/somewhere/in/your/PATH")
   ```
   Then, to execute Trixi.jl in parallel, execute the following command from your
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
   get the hang of it, developing Trixi.jl in parallel becomes much smoother this
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
For information on how to evaluate the parallel performance of Trixi.jl, please
have a look at the [Performance metrics of the `AnalysisCallback`](@ref performance-metrics)
section, specifically at the descriptions of the performance index (PID).


### Using error-based step size control with MPI
If you use error-based step size control (see also the section on
[error-based adaptive step sizes](@ref adaptive_step_sizes)) together with MPI you need to pass
`internalnorm=ode_norm` and you should pass `unstable_check=ode_unstable_check` to
OrdinaryDiffEq's [`solve`](https://docs.sciml.ai/DiffEqDocs/latest/basics/common_solver_opts/),
which are both included in [`ode_default_options`](@ref).

### Using parallel input and output
Trixi.jl allows parallel I/O using MPI by leveraging parallel HDF5.jl. On most systems, this is
enabled by default. Additionally, you can also use a local installation of the HDF5 library
(with MPI support). For this, you first need to use a system-provided MPI library, see also
[here](@ref parallel_system_MPI) and you need to tell [HDF5.jl](https://github.com/JuliaIO/HDF5.jl)
to use this library. To do so with HDF5.jl v0.17 and newer, set the preferences `libhdf5` and
`libhdf5_hl` to the local paths of the libraries `libhdf5` and `libhdf5_hl`, which can be done by
```julia
julia> using Preferences, UUIDs
julia> set_preferences!(
           UUID("f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"), # UUID of HDF5.jl
           "libhdf5" => "/path/to/your/libhdf5.so",
           "libhdf5_hl" => "/path/to/your/libhdf5_hl.so", force = true)
```
Alternatively, with HDF5.jl v0.17.1 or higher you can use
```julia
julia> using HDF5
julia> HDF5.API.set_libraries!("/path/to/your/libhdf5.so", "/path/to/your/libhdf5_hl.so")
```
For more information see also the
[documentation of HDF5.jl](https://juliaio.github.io/HDF5.jl/stable/mpi/). In total, you should
have a file called `LocalPreferences.toml` in the project directory that contains a section
`[MPIPreferences]`, a section `[HDF5]` with entries `libhdf5` and `libhdf5_hl`, a section `[P4est]`
with the entry `libp4est` as well as a section `[T8code]` with the entries `libt8`, `libp4est`
and `libsc`.
If you use HDF5.jl v0.16 or older, instead of setting the preferences for HDF5.jl, you need to set
the environment variable `JULIA_HDF5_PATH` to the path, where the HDF5 binaries are located and
then call `]build HDF5` from Julia.

If HDF5 is not MPI-enabled, Trixi.jl will fall back on a less efficient I/O mechanism. In that
case, all disk I/O is performed only on rank zero and data is distributed to/gathered from the
other ranks using regular MPI communication.
