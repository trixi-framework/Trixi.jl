# [Time integration methods](@id time-integration)

Trixi.jl is compatible with the [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/).
In particular, explicit Runge-Kutta methods from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
are tested extensively.
Interesting classes of time integration schemes are
- [Explicit low-storage Runge-Kutta methods](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Low-Storage-Methods)
- [Strong stability preserving methods](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws))

Some common options for `solve` from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
are the following. Further documentation can be found in the
[SciML docs](https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/).
- If you use a fixed time step method like `CarpenterKennedy2N54`, you need to pass
  a time step as `dt=...`. If you use a [`StepsizeCallback`](@ref), the value passed 
  as `dt=...` is irrelevant since it will be overwritten by the [`StepsizeCallback`](@ref).
  If you want to use an adaptive time step method such as `SSPRK43` or `RDPK3SpFSAL49`
  and still want to use CFL-based step size control via the [`StepsizeCallback`](@ref),
  you need to pass the keyword argument `adaptive=false` to `solve`.
- You should usually set `save_everystep=false`. Otherwise, OrdinaryDiffEq.jl will
  (try to) save the numerical solution after every time step in RAM (until you run
  out of memory or start to swap).
- You can set the maximal number of time steps via `maxiters=...`.
- SSP methods and many low-storage methods from OrdinaryDiffEq.jl support
  `stage_limiter!`s and `step_limiter!`s, e.g., [`PositivityPreservingLimiterZhangShu`](@ref)
  from Trixi.jl.
- If you start Julia with multiple threads and want to use them also in the time 
  integration method from OrdinaryDiffEq.jl, you need to pass the keyword argument
  `thread=OrdinaryDiffEq.True()` to the algorithm, e.g., 
  `RDPK3SpFSAL49(thread=OrdinaryDiffEq.True())` or 
  `CarpenterKennedy2N54(thread=OrdinaryDiffEq.True(), williamson_condition=false)`.
  For more information on using thread-based parallelism in Trixi.jl, please refer to 
  [Shared-memory parallelization with threads](@ref).
- If you use error-based step size control (see also the section on
  [error-based adaptive step sizes](@ref adaptive_step_sizes) together with MPI, you need to
  pass `internalnorm=ode_norm` and you should pass `unstable_check=ode_unstable_check` to
  OrdinaryDiffEq's [`solve`](https://docs.sciml.ai/DiffEqDocs/latest/basics/common_solver_opts/), which are both
  included in [`ode_default_options`](@ref).

!!! note "Number of `rhs!` calls"
    If you use explicit Runge-Kutta methods from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl),
    the total number of `rhs!` calls can be (slightly) bigger than the number of steps times the number
    of stages, e.g. to allow for interpolation (dense output), root-finding for continuous callbacks,
    and error-based time step control. In general, you often should not need to worry about this if you
    use Trixi.jl.
