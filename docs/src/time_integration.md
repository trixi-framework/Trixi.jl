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

## Optimized Schemes

Optimized schemes aim to maximize the stability region or to tailor the stability polynomial to specific problems, such as stiff equations. By optimizing the stability polynomial, these schemes can achieve greater accuracy and efficiency. One of the optimized schemes that is implemented in Trixi.jl is the Paired Explicit Runge-Kutta method.

### Paired explicit Runge-Kutta schemes Schemes

Paired Explicit Runge-Kutta (PERK) or `PairedExplicitRK` schemes are an advanced class of numerical methods designed to efficiently solve ordinary differential equations (ODEs). They work by pairing stages in the Runge-Kutta process, reducing redundant computations and minimizing storage requirements while maintaining high-order accuracy. The stability polynomial in PERK schemes is optimized to allow for larger time steps, making them particularly effective in handling mildly stiff systems where traditional explicit methods would struggle. This combination of efficiency, reduced computational cost, and enhanced stability makes PERK schemes a powerful tool for solving ODEs in scenarios where both precision and performance are critical. In this type of schemes, additional libraries have to be imported to perform the aforementioned optimization.

### Tutorial: Using `PairedExplicitRK2`

In this following tutorial, we will demonstrate how you can use the second order paired explicit Runge-Kutta schemes time integrator.

1. First, ensure you have the necessary packages installed. For the paired explicit Runge-Kutta scheme of the second order, you need an additional package of `Convex.jl` and `ECOS.jl`. You can install them using Julia's package manager:

```julia
using Pkg
Pkg.add("Trixi")
Pkg.add("Convex")
Pkg.add("ECOS")
```

2. In order to use the time integrator, you also need to import these packages in the script you are running as well:

```julia
using Convex, ECOS
using Trixi
```

3. Define the ODE problem and the semidiscretization setup. For this example, we will use a simple advection problem.

```julia
# Define the mesh
cells_per_dimension = 100
coordinates_min = 0.0
coordinates_max = 1.0
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)


# Define the equations and initial condition
equations = LinearScalarAdvectionEquation()
initial_condition = (x, t) -> sin(2π * x)

# Define the solver
solver = FluxBasedSolver()

# Define the semidiscretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
```

4. Define the necessary callbacks for the simulation. Callbacks are used to perform actions at specific points during the integration process.

```julia
# Define the callbacks
summary_callback = SummaryCallback()
alive_callback = AliveCallback()
save_solution = SaveSolutionCallback(dt = 0.1, save_initial_solution = true, save_final_solution = true)
analysis_callback = AnalysisCallback(semi, interval = 200)
stepsize_callback = StepsizeCallback(cfl = 3.7)

# Create a CallbackSet to collect all callbacks
callbacks = CallbackSet(summary_callback, alive_callback, save_solution, analysis_callback, stepsize_callback)
```

5. Define the ODE problem by specifying the time span over which the ODE will be solved. The `tspan` parameter is a tuple `(t_start, t_end)` that defines the start and end times for the simulation. The `semidiscretize` function is used to create the ODE problem from the semidiscretization setup.

```julia
# Define the time span
tspan = (0.0, 1.0)

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, tspan)
```

6. In this step we will construct the time integrator. In order to do this, you need the following components:

  - Number of Stages: The number of stages in the Runge-Kutta method. In this example, we use `6` stages.
  - Time Span (`tspan`): A tuple `(t_start, t_end)` that defines the time span over which the ODE will be solved. This is used to calculate the maximum time step allowed for the bisection algorithm used in calculating the polynomial coefficients in the ODE algorithm. This variable is already defined in step 5.
  - Semidiscretization (`semi`): The semidiscretization setup that includes the mesh, equations, initial condition, and solver. In this example, this variable is already defined in step 3.

```julia
# Construct second order paired explicit Runge-Kutta method with 6 stages for given simulation setup.
# Pass `tspan` to calculate maximum time step allowed for the bisection algorithm used 
# in calculating the polynomial coefficients in the ODE algorithm.
ode_algorithm = Trixi.PairedExplicitRK2(6, tspan, semi)
```

7. With everything now set up, you can now use `Trixi.solve` to solve the ODE problem. The `solve` function takes the ODE problem, the time integrator, and some options such as the time step (`dt`), whether to save every step (`save_everystep`), and the callbacks.

```julia
# Solve the ODE problem using PERK2
sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks)
```