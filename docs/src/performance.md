# Performance

Trixi.jl is designed to balance performance and readability. Since Julia provides
a lot of zero-cost abstractions, it is often possible to optimize both goals
simultaneously.

The usual development workflow in Julia is

1. Make it work.
2. Make it nice.
3. Make it fast.

To achieve the third step, you should be familiar with (at least) the section on
[performance tips in the Julia manual](https://docs.julialang.org/en/v1/manual/performance-tips/).
Here, we just list some important aspects you should consider when developing Trixi.jl.

- Consider using `@views`/`view(...)` when using array slices, except on the left-side
  of an assignment
  ([further details](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-views)).
- Functions are essentially for free, since they are usually automatically inlined where it makes sense (using `@inline` can be used as an additional hint to the compiler)
  ([further details](https://docs.julialang.org/en/v1/manual/performance-tips/#Break-functions-into-multiple-definitions)).
- Function barriers can improve performance due to type stability
  ([further details](https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions)).
- Look for type instabilities using `@code_warntype`.
  Consider using `@descend` from [Cthulhu.jl](https://github.com/JuliaDebug/Cthulhu.jl) to investigate
  deeper call chains.



## Manual benchmarking

If you modify some internal parts of Trixi.jl, you should check the impact on performance.
Hence, you should at least investigate the performance roughly by comparing the reported
timings of several elixirs. Deeper investigations and micro-benchmarks should usually use
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
For example, the following steps were used to benchmark the changes introduced in
[PR #256](https://github.com/trixi-framework/Trixi.jl/pull/256).

1. `git checkout e7ebf3846b3fd62ee1d0042e130afb50d7fe8e48` (new version)
2. Start `julia --threads=1 --check-bounds=no`.
3. Execute the following code in the REPL to benchmark the `rhs!` call at the final state.
   ```julia
   julia> using BenchmarkTools, Revise; using Trixi

   julia> # nowadays "examples/tree_2d_dgsem/elixir_euler_sedov_blast_wave.jl"
          trixi_include("examples/2d/elixir_euler_sedov_blast_wave.jl")

   julia> du_test = copy(sol.u[end]); u_test = copy(sol.u[end]);

   julia> @benchmark Trixi.rhs!(
             $(du_test),
             $(u_test),
             $(semi),
             $(sol.t[end]))
   BenchmarkTools.Trial:
    memory estimate:  10.48 KiB
    allocs estimate:  67
    --------------
    minimum time:     4.510 ms (0.00% GC)
    median time:      4.646 ms (0.00% GC)
    mean time:        4.699 ms (0.00% GC)
    maximum time:     7.183 ms (0.00% GC)
    --------------
    samples:          1065
    evals/sample:     1

   shell> git checkout 222241ff54f8a4ca9876cc1fc25ae262416a4ea0

   julia> # nowadays "examples/tree_2d_dgsem/elixir_euler_sedov_blast_wave.jl"
          trixi_include("examples/2d/elixir_euler_sedov_blast_wave.jl")

   julia> @benchmark Trixi.rhs!(
             $(du_test),
             $(u_test),
             $(semi),
             $(sol.t[end]))
   BenchmarkTools.Trial:
    memory estimate:  10.36 KiB
    allocs estimate:  67
    --------------
    minimum time:     4.500 ms (0.00% GC)
    median time:      4.635 ms (0.00% GC)
    mean time:        4.676 ms (0.00% GC)
    maximum time:     5.880 ms (0.00% GC)
    --------------
    samples:          1070
    evals/sample:     1
   ```
   Run the `@benchmark ...` commands multiple times to see whether there are any significant fluctuations.
   Note that the elixir name has changed since
   [PR #256](https://github.com/trixi-framework/Trixi.jl/pull/256).
   Nowadays, the relevant elixir is
   [`examples/tree_2d_dgsem/elixir_euler_sedov_blast_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_sedov_blast_wave.jl).

Follow these steps for both commits you want to compare. The relevant benchmark results you should typically be looking at
are the median and mean values of the runtime and the memory/allocs estimate. In this example, the differences
of the runtimes are of the order of the fluctuations one gets when running the benchmarks multiple times. Since
the memory/allocs are (roughly) the same, there doesn't seem to be a significant performance regression here.

You can also make it more detailed by benchmarking only, e.g., the calculation of the volume terms, but whether that's necessary depends on the modifications you made and their (potential) impact.

Some more detailed description of manual profiling and benchmarking as well as
resulting performance improvements of Trixi.jl are given in the following blog posts.
- [Improving performance of AMR with p4est](https://ranocha.de/blog/Optimizing_p4est_AMR/),
  cf. [#638](https://github.com/trixi-framework/Trixi.jl/pull/638)
- [Improving performance of EC methods](https://ranocha.de/blog/Optimizing_EC_Trixi/),
  cf. [#643](https://github.com/trixi-framework/Trixi.jl/pull/643)


## Automated benchmarking

We use [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl) to provide a standard set of
benchmarks for Trixi.jl. The relevant benchmark script is
[benchmark/benchmarks.jl](https://github.com/trixi-framework/Trixi.jl/blob/main/benchmark/benchmarks.jl).
To benchmark the changes made in a PR, please proceed as follows:

1. Check out the latest `main` branch of your Trixi.jl development repository.
2. Check out the latest development branch of your PR.
3. Change your working directory to the `benchmark` directory of Trixi.jl.
4. Execute `julia run_benchmarks.jl`.

This will take some hours to complete and requires at least 8 GiB of RAM. When everything is finished, some
output files will be created in the `benchmark` directory of Trixi.jl.

!!! warning
    Please note that the benchmark scripts use `--check-bounds=no` at the moment.
    Thus, they will not work in any useful way for Julia v1.10 (and newer?), see
    [Julia issue #50985](https://github.com/JuliaLang/julia/issues/50985).

You can also run a standard set of benchmarks manually via
```julia
julia> using PkgBenchmark, Trixi

julia> results = benchmarkpkg(Trixi, BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=1`))

julia> export_markdown(pkgdir(Trixi, "benchmark", "single_benchmark.md"), results)
```
This will save a markdown file with a summary of the benchmark results similar to
[this example](https://gist.github.com/ranocha/494fa2529e1e6703c17b08434c090980).
Note that this will take quite some time. Additional options are described in the
[docs of PkgBenchmark.jl](https://juliaci.github.io/PkgBenchmark.jl/stable).
A particularly useful option is to specify a `BenchmarkConfig` including Julia
command line options affecting the performance such as disabling bounds-checking
and setting the number of threads.

A useful feature when developing Trixi.jl is to compare the performance of Trixi.jl's
current state vs. the `main` branch. This can be achieved by executing
```julia
julia> using PkgBenchmark, Trixi

julia> results = judge(Trixi,
             BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=1`), # target
             BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=1`, id="main") # baseline
       )

julia> export_markdown(pkgdir(Trixi, "benchmark", "results.md"), results)
```
By default, the `target` is the current state of the repository. Remember that you
need to be in a clean state (commit or stash your changes) to run this successfully.
You can also run this comparison and an additional one using two threads via
```julia
julia> include("benchmark/run_benchmarks.jl")
```
Then, markdown files including the results are saved in `benchmark/`.
[This example result](https://gist.github.com/ranocha/bf98d19e288e759d3a36ca0643448efb)
was obtained using a GitHub action for the
[PR #535](https://github.com/trixi-framework/Trixi.jl/pull/535).
Note that GitHub actions run on in the cloud in a virtual machine. Hence, we do not really
have control over it and performance results must be taken with a grain of salt.
Nevertheless, significant runtime differences and differences of memory allocations
should be robust indicators of performance changes.


## Runtime performance vs. latency aka using `@nospecialize` selectively

Usually, Julia will compile specialized versions of each method, using as much information from the
types of function arguments as possible (based on some
[heuristics](https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing)).
The compiler will generate code that is as efficient as comparable code written in a low-level
language such as C or Fortran. However, there are cases where the runtime performance does not
really matter but the time needed to compile specializations becomes significant. This is related to
latency or the time-to-first-plot problem, well-known in the Julia community. In such a case, it can
be useful to remove some burden from the compiler by avoiding specialization on every possible argument
types using [the macro `@nospecialize`](https://docs.julialang.org/en/v1/base/base/#Base.@nospecialize).
A prime example of such a case is pretty printing of `struct`s in the Julia REPL, see the
[associated PR](https://github.com/trixi-framework/Trixi.jl/pull/447) for further discussions.

As a rule of thumb:
- Do not use `@nospecialize` in performance-critical parts, in particular not for methods involved
  in computing `Trixi.rhs!`.
- Consider using `@nospecialize` for methods like custom implementations of `Base.show`.


## [Performance metrics of the `AnalysisCallback`](@id performance-metrics)
The [`AnalysisCallback`](@ref) computes two performance indicators that you can use to
evaluate the serial and parallel performance of Trixi.jl. They represent
measured run times that are normalized by the number of `rhs!` evaluations and
the number of degrees of freedom of the problem setup. The normalization ensures that we can
compare different measurements for each type of indicator independent of the number of
time steps or mesh size. All indicators have in common that they are still in units of
time, thus *lower is better* for each of them.

Here, the term "degrees of freedom" (DOFs) refers to the number of *independent*
state vectors that are used to represent the numerical solution. For example, if
you use a DGSEM-type scheme in 2D on a mesh with 8 elements and with
5-by-5 Gauss-Lobatto nodes in each element (i.e., a polynomial degree of 4), the
total number of DOFs would be
```math
n_\text{DOFs,DGSEM} = \{\text{number of elements}\} \cdot \{\text{number of nodes per element}\} = 8 \cdot (5 \cdot 5) = 200.
```
In contrast, for a finite volume-type scheme on a mesh with 8 elements, the total number of
DOFs would be (independent of the number of spatial dimensions)
```math
n_\text{DOFs,FV} = \{\text{number of elements}\} = 8,
```
since for standard finite volume methods you store a single state vector in each
element. Note that we specifically count the number of state *vectors* and not
the number of state *variables* for the DOFs. That is, in the previous example
``n_\text{DOFs,FV}`` is equal to 8 independent of whether this is a compressible Euler
setup with 5 state variables or a linear scalar advection setup with one state
variable.

For each indicator, the measurements are always since the last invocation of the
`AnalysisCallback`. That is, if the analysis callback is called multiple times,
the indicators are repeatedly computed and can thus also be used to track the
performance over the course of a longer simulation, e.g., to analyze setups with varying performance
characteristics. Note that the time spent in the `AnalysisCallback` itself is always
*excluded*, i.e., the performance measurements are not distorted by potentially
expensive solution analysis computations. All other parts of a Trixi.jl simulation
are included, however, thus make sure that you disable everything you do *not*
want to be measured (such as I/O callbacks, visualization etc.).

!!! note "Performance indicators and adaptive mesh refinement"
    Currently it is not possible to compute meaningful performance indicators for a simulation
    with arbitrary adaptive mesh refinement, since this would require to
    explicitly keep track of the number of DOF updates due to the mesh size
    changing repeatedly. The only way to do this at the moment is by setting the
    analysis interval to the same value as the AMR interval.

### Local, `rhs!`-only indicator
The *local, `rhs!`-only indicator* is computed as
```math
\text{time/DOF/rhs!} = \frac{t_\text{\texttt{rhs!}}}{n_\text{DOFs,local} \cdot n_\text{calls,\texttt{rhs!}}},
```
where ``t_\text{\texttt{rhs!}}`` is the accumulated time spent in `rhs!`,
``n_\text{DOFs,local}`` is the *local* number of DOFs (i.e., on the
current MPI rank; if doing a serial run, you can just think of this as *the*
number of DOFs), and ``n_\text{calls,\texttt{rhs!}}`` is the number of times the
`rhs!` function has been evaluated. Note that for this indicator, we measure *only*
the time spent in `rhs!`, i.e., by definition all computations outside of `rhs!` - specifically
all other callbacks and the time integration method - are not taken into account.

The local, `rhs!`-only indicator is usually most useful if you do serial
measurements and are interested in the performance of the implementation of your
core numerical methods (e.g., when doing performance tuning).

### Performance index (PID)
The *performance index* (PID) is computed as
```math
\text{PID} = \frac{t_\text{wall} \cdot n_\text{ranks,MPI}}{n_\text{DOFs,global} \cdot n_\text{calls,\texttt{rhs!}}},
```
where ``t_\text{wall}`` is the walltime since the last call to the `AnalysisCallback`,
``n_\text{ranks,MPI}`` is the number of MPI ranks used,
``n_\text{DOFs,global}`` is the *global* number of DOFs (i.e., the sum of
DOFs over all MPI ranks; if doing a serial run, you can just think of this as *the*
number of DOFs), and ``n_\text{calls,\texttt{rhs!}}`` is the number of times the
`rhs!` function has been evaluated since the last call to the `AnalysisCallback`.
The PID measures everything except the time spent in the `AnalysisCallback` itself -
specifically, all other callbacks and the time integration method itself are included.

The PID is usually most useful if you would like to compare the
parallel performance of your code to its serial performance. Specifically, it
allows you to evaluate the parallelization overhead of your code by giving you a
measure of the resources that are necessary to solve a given simulation setup.
In a sense, it mimics the "core hours" metric that is often used by
supercomputer centers to measure how many resources a particular compute job
requires. It can thus be seen as a proxy for "energy used" and, as an extension, "monetary cost".

!!! note "Initialization overhead in measurements"
    When using one of the integration schemes from OrdinaryDiffEq.jl, their implementation
    will initialize some OrdinaryDiffEq.jl-specific information during the first
    time step. Among other things, one additional call to `rhs!` is performed.
    Therefore, make sure that for performance measurements using the PID either the
    number of timesteps or the workload per `rhs!` call is large enough to make
    the initialization overhead negligible. Note that the extra call to `rhs!`
    is properly accounted for in both the number of calls and the measured time,
    so you do not need to worry about it being expensive. If you want a perfect
    timing result, you need to set the analysis interval such that the
    `AnalysisCallback` is invoked at least once during the course of the simulation and
    discard the first PID value.
