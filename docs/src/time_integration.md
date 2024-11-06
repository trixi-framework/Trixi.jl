# [Time integration methods](@id time-integration)

## Methods from OrdinaryDiffEq.jl

Trixi.jl is compatible with the [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/).
In particular, [explicit Runge-Kutta methods](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Explicit-Runge-Kutta-Methods) from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
are tested extensively.
Interesting classes of time integration schemes are
- [Explicit low-storage Runge-Kutta methods](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Low-Storage-Methods)
- [Strong stability preserving (SSP) methods](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws))

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
  `stage_limiter!`s and `step_limiter!`s, e.g., [`PositivityPreservingLimiterZhangShu`](@ref) and [`EntropyBoundedLimiter`](@ref)
  from Trixi.jl.
- If you start Julia with multiple threads and want to use them also in the time 
  integration method from OrdinaryDiffEq.jl, you need to pass the keyword argument
  `thread=OrdinaryDiffEq.True()` to the algorithm, e.g., 
  `RDPK3SpFSAL49(thread=OrdinaryDiffEq.True())` or 
  `CarpenterKennedy2N54(thread=OrdinaryDiffEq.True(), williamson_condition=false)`.
  For more information on using thread-based parallelism in Trixi.jl, please refer to 
  [Shared-memory parallelization with threads](@ref).
- If you use error-based step size control (see also the section on
  [error-based adaptive step sizes](@ref adaptive_step_sizes)) together with MPI, you need to
  pass `internalnorm=ode_norm` and you should pass `unstable_check=ode_unstable_check` to
  OrdinaryDiffEq's [`solve`](https://docs.sciml.ai/DiffEqDocs/latest/basics/common_solver_opts/), which are both
  included in [`ode_default_options`](@ref).

!!! note "Number of `rhs!` calls"
    If you use explicit Runge-Kutta methods from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl),
    the total number of `rhs!` calls can be (slightly) bigger than the number of steps times the number
    of stages, e.g. to allow for interpolation (dense output), root-finding for continuous callbacks,
    and error-based time step control. In general, you often should not need to worry about this if you
    use Trixi.jl.

## Custom Optimized Schemes

### Stabilized Explicit Runge-Kutta Methods

Optimized explicit schemes aim to maximize the timestep $\Delta t$ for a given simulation setup.
Formally, this boils down to an optimization problem of the form 
```math
\underset{P_{p;S} \, \in \, \mathcal{P}_{p;S}}{\max} \Delta t \text{ such that } \big \vert P_{p;S}(\Delta t \lambda_m) \big \vert \leq 1, \quad  m = 1 , \dots , M \tag{1}
```
where $p$ denotes the order of consistency of the scheme, $S$ is the number of stage evaluations and $M$ denotes the number of eigenvalues $\lambda_m$ of the Jacobian matrix $J \coloneqq \frac{\partial \boldsymbol F}{\partial \boldsymbol U}$ of the right-hand side of the [semidiscretized PDE](https://trixi-framework.github.io/Trixi.jl/stable/overview/#overview-semidiscretizations):
```math
\dot{\boldsymbol U} = \boldsymbol F(\boldsymbol U) \tag{2} \: .
```
In particular, for $S > p$ the Runge-Kutta method includes some free coefficients which may be used to adapt the domain of absolute stability to the problem at hand.
Since Trixi.jl [supports exact computation of the Jacobian $J$ by means of automatic differentiation](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/differentiable_programming/), we have access to the Jacobian of a given simulation setup.
For small (up to approximately $10^4$ DoF) systems, the spectrum $\boldsymbol \sigma = \left \{ \lambda_m \right \}_{m=1, \dots, M}$ can be computed directly using [`LinearAlgebra.eigvals(J)`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigvals).
For larger systems, we recommend the procedure outlined in section 4.1 of [Doehring et al. (2024)](https://doi.org/10.1016/j.jcp.2024.113223). This approach computes a reduced set of (estimated) eigenvalues $\widetilde{\boldsymbol \sigma}$ around the convex hull of the spectrum by means of the [Arnoldi method](https://github.com/JuliaLinearAlgebra/Arpack.jl).

The optimization problem (1) can be solved using the algorithms described in [Ketcheson, Ahmadia (2012)](http://dx.doi.org/10.2140/camcos.2012.7.247) for a moderate number of stages $S$ or [Doehring, Gassner, Torrilhon (2024)](https://doi.org/10.1007/s10915-024-02478-5) for a large number of stages $S$.
In Trixi.jl, the former approach is implemented by means of convex optimization using the [Convex.jl](https://github.com/jump-dev/Convex.jl) package.

The resulting stability polynomial $P_{p;S}$ is then used to construct an optimized Runge-Kutta method.
Trixi.jl implements the [Paired-Explicit Runge-Kutta (P-ERK)](https://doi.org/10.1016/j.jcp.2019.05.014) method, a low-storage, multirate-ready method with optimized domain of absolute stability.

### Paired-Explicit Runge-Kutta (P-ERK) Schemes

Paired-Explicit Runge-Kutta (P-ERK) or `PairedExplicitRK` schemes are an advanced class of numerical methods designed to efficiently solve ODEs.
In the [original publication](https://doi.org/10.1016/j.jcp.2019.05.014), second-order schemes were introduced, which have been extended to [third](https://doi.org/10.1016/j.jcp.2022.111470)- and [fourth](https://doi.org/10.48550/arXiv.2408.05470)-order in subsequent works.

By construction, P-ERK schemes are suited for integrating multirate systems, i.e., systems with varying characteristics speeds throughout the domain.
Nevertheless, due to their optimized stability properties and low-storage nature, the P-ERK schemes are also highly efficient when applied as standalone methods.

#### Tutorial: Using `PairedExplicitRK2`

In this tutorial, we will demonstrate how you can use the second-order P-ERK time integrator.

1. First, ensure you have the necessary packages installed. For the optimization of the stability polynomial $P_{2; S}$, you need the `Convex.jl` and `ECOS.jl` packages.
You can install them using Julia's package manager:

```julia
using Pkg
Pkg.add("Trixi")
Pkg.add("Convex")
Pkg.add("ECOS")
```

2. Now you can load the necessary packages:

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
mesh = StructuredMesh(cells_per_dimension, 
                      coordinates_min, coordinates_max)

# Define the equation and initial condition
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

initial_condition = initial_condition_convergence_test

# Define the solver
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Define the semidiscretization
semi = SemidiscretizationHyperbolic(mesh, 
                                    equations, initial_condition, 
                                    solver)
```

4. Define the necessary callbacks for the simulation. Callbacks are used to perform actions at specific points during the integration process.

```julia
# Define some standard callbacks
summary_callback  = SummaryCallback()
alive_callback    = AliveCallback()
analysis_callback = AnalysisCallback(semi, interval = 200)
# For this optimized method we can use a relatively large CFL number
stepsize_callback = StepsizeCallback(cfl = 2.5)

# Create a CallbackSet to collect all callbacks
callbacks = CallbackSet(summary_callback, 
                        alive_callback, 
                        analysis_callback, 
                        stepsize_callback)
```

5. Define the ODE problem by specifying the time span over which the ODE will be solved. 
The `tspan` parameter is a tuple `(t_start, t_end)` that defines the start and end times for the simulation. 
The `semidiscretize` function is used to create the ODE problem from the simulation setup.

```julia
# Define the time span
tspan = (0.0, 1.0)

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, tspan)
```

6. In this step we will construct the time integrator. In order to do this, you need the following components:

  - Number of Stages: The number of stages $S$ in the Runge-Kutta method. 
  In this example, we use `6` stages.
  - Time Span (`tspan`): A tuple `(t_start, t_end)` that defines the time span over which the ODE will be solved. 
  This defines the bounds for the bisection routine for the optimal timestep $\Delta t$ used in calculating the polynomial coefficients at optimization stage. 
  This variable is already defined in step 5.
  - Semidiscretization (`semi`): The semidiscretization setup that includes the mesh, equations, initial condition, and solver. In this example, this variable is already defined in step 3.
  In the background, we compute from `semi` the Jacobian $J$ evaluated at the initial condition using [`jacobian_ad_forward`](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.jacobian_ad_forward-Tuple{Trixi.AbstractSemidiscretization}).
  This is then followed by the computation of the spectrum $\boldsymbol \sigma(J)$ using `LinearAlgebra.eigvals`.
  Equipped with the spectrum, the optimal stability polynomial is computed, from which the corresponding Runge-Kutta method is derived. Other constructors (if the coefficients $\boldsymbol{\alpha}$ of the stability polynomial are already available, or if a reduced spectrum $\widetilde{\boldsymbol{\sigma}}$ should be used) are discussed below.

```julia
# Construct second order-explicit Runge-Kutta method with 6 stages for given simulation setup (`semi`)
# `tspan` provides the bounds for the bisection routine that is used to calculate the maximum timestep.
ode_algorithm = Trixi.PairedExplicitRK2(6, tspan, semi)
```

7. With everything now set up, you can now use `Trixi.solve` to solve the ODE problem. The `solve` function takes the ODE problem, the time integrator, and some options such as the time step (`dt`), whether to save every step (`save_everystep`), and the callbacks.

```julia
# Solve the ODE problem using PERK2
sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # overwritten by `stepsize_callback`
                  save_everystep = false, callback = callbacks)
```

8. Advanced constructors:
There are two additional constructors for the `PairedExplicitRK2` method besides the one taking in a semidiscretization `semi`:
  - `PairedExplicitRK2(num_stages, base_path_monomial_coeffs::AbstractString)` constructs a `num_stages`-stage method from the given optimal monomial_coeffs $\boldsymbol \alpha$.
  These are expected to be present in the provided directory in the form of a `gamma_<S>.txt` file, where `<S>` is the number of stages `num_stages`.
  This constructor is useful when the optimal coefficients cannot be obtained using the optimization routine by Ketcheson and Ahmadia, possibly due to a large number of stages $S$.
  - `PairedExplicitRK2(num_stages, tspan, eig_vals::Vector{ComplexF64})` constructs a `num_stages`-stage using the optimization approach by Ketcheson and Ahmadia for the (reduced) spectrum `eig_vals`.
  The use-case for this constructor would be a large system, for which the computation of all eigenvalues is infeasible.

#### Automatic computation of stable CFL Number

In the previous tutorial the CFL number was set manually to $2.5$.
To avoid this trial-and error process, instantiations of `AbstractPairedExplicitRK` methods can automatically compute the stable CFL number for a given simulation setup using the [`Trixi.calculate_cfl`](@ref) function.
When constructing the time integrator from a semidiscretization `semi`, 
```julia
# Construct third-order paired-explicit Runge-Kutta method with 8 stages for given simulation setup.                               
ode_algorithm = Trixi.PairedExplicitRK3(8, tspan, semi)
```
the maximum timestep `dt` is stored by the `ode_algorithm`.
This can then be used to compute the stable CFL number for the given simulation setup:
```julia
cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
```
For nonlinear problems, the spectrum will in general change over the course of the simulation.
Thus, it is often necessary to reduce the optimal `cfl_number` by a safety factor:
```julia
# For non-linear problems, the CFL number should be reduced by a safety factor
# as the spectrum changes (in general) over the course of a simulation
stepsize_callback = StepsizeCallback(cfl = 0.85 * cfl_number)
```
If the optimal monomial coefficients are precomputed, the user needs to set the obtained maximum timestep from the optimization manually via
```julia
ode_algorithm.dt_opt = 42.0 # The timestep obtained from the optimization
```
Then, the stable CFL number can be computed as described above.

#### Currently implemented P-ERK methods

##### Single/Standalone methods

- [`Trixi.PairedExplicitRK2`](@ref): Second-order P-ERK method with at least two stages.
- [`Trixi.PairedExplicitRK3`](@ref): Third-order P-ERK method with at least three stages.