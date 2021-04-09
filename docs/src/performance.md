# Performance

Trixi.jl is designed to balance performance and readability. Since Julia provides
a lot of zero-cost abstractions, it is often possible to optimize both goals
simultaneously.

The usual development workflow in Julia is

1. Make it work.
2. Make it fast.

To achieve the second step, you should be familiar with (at least) the section on
[performance tips in the Julia manual](https://docs.julialang.org/en/v1/manual/performance-tips/).
Here, we just list some important aspects you should consider when developing Trixi.

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

If you modify some internal parts of Trixi, you should check the impact on performance.
Hence, you should at least investigate the performance roughly by comparing the reported
timings of several elixirs. Deeper investigations and micro-benchmarks should usually use
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
For example, the following steps were used to benchmark the changes introduced in
https://github.com/trixi-framework/Trixi.jl/pull/256.

1. `git checkout e7ebf3846b3fd62ee1d0042e130afb50d7fe8e48` (new version)
2. Start `julia --threads=1 --check-bounds=no`.
3. Execute the following code in the REPL to benchmark the `rhs!` call at the final state.
   ```julia
   julia> using BenchmarkTools, Revise; using Trixi

   julia> trixi_include("examples/2d/elixir_euler_sedov_blast_wave.jl")

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

   julia> trixi_include("examples/2d/elixir_euler_sedov_blast_wave.jl")

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

Follow these steps for both commits you want to compare. The relevant benchmark results you should typically be looking at
are the median and mean values of the runtime and the memory/allocs estimate. In this example, the differences
of the runtimes are of the order of the fluctuations one gets when running the benchmarks multiple times. Since
the memory/allocs are (roughly) the same, there doesn't seem to be a significant performance regression here.

You can also make it more detailed by benchmarking only, e.g., the calculation of the volume terms, but whether that's necessary depends on the modifications you made and their (potential) impact.


## Automated benchmarking

We use [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl) to provide a standard set of
benchmarks for Trixi.



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
