# [Callbacks](@id callbacks-id)
Many of the advanced features of Trixi.jl, such as adaptive mesh refinement, are implemented as
callbacks. A callback is an algorithmic entity that gets passed to the ODE solver and
is called at specific points during execution to perform certain tasks. Callbacks in Trixi.jl are
either called after each time step (*step callbacks*) or after each stage of the ODE
solver (*stage callbacks*).

![callbacks_illustration](https://user-images.githubusercontent.com/65298011/108088616-f690c000-7078-11eb-9dd1-b673eac6cecf.png)

The advantage of callbacks over hard-coding all features is that it allows to extend Trixi.jl without
modifying the internal source code. Trixi.jl provides callbacks for time step
control, adaptive mesh refinement, I/O, and more.

## Step callbacks

### CFL-based time step control
Time step control can be performed with a [`StepsizeCallback`](@ref). An example making use
of this can be found at [`examples/tree_2d_dgsem/elixir_advection_basic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_basic.jl)

### Adaptive mesh refinement
Trixi.jl uses a hierarchical Cartesian mesh which can be locally refined in a solution-adaptive way.
This can be used to speed up simulations with minimal loss in overall accuracy. Adaptive mesh refinement (AMR) can be used by
passing an [`AMRCallback`](@ref) to the ODE solver. The `AMRCallback` requires a controller such as
[`ControllerThreeLevel`](@ref) or [`ControllerThreeLevelCombined`](@ref) to tell the AMR
algorithm which cells to refine/coarsen.

An example elixir using AMR can be found at [`examples/tree_2d_dgsem/elixir_advection_amr.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_amr.jl).

### Analyzing the numerical solution
The [`AnalysisCallback`](@ref) can be used to analyze the numerical solution, e.g. calculate
errors or user-specified integrals, and print the results to the screen. The results can also be
saved in a file. An example can be found at [`examples/tree_2d_dgsem/elixir_euler_vortex.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_vortex.jl).
Note that the errors (e.g. `L2 error` or `Linf error`) are computed with respect to the initial condition.
The percentage of the simulation time refers to the ratio of the current time and the final time, i.e. it does
not consider the maximal number of iterations. So the simulation could finish before 100% are reached.
Note that, e.g., due to AMR or smaller time step sizes, the simulation can actually take longer than
the percentage indicates.
In [Performance metrics of the `AnalysisCallback`](@ref performance-metrics) you can find a detailed
description of the different performance metrics the `AnalysisCallback` computes.

### I/O

#### Solution and restart files
To save the solution in regular intervals you can use a [`SaveSolutionCallback`](@ref). It is also
possible to create restart files using the [`SaveRestartCallback`](@ref). An example making use
of these can be found at [`examples/tree_2d_dgsem/elixir_advection_extended.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_extended.jl).
An example showing how to restart a simulation from a restart file can be found at
[`examples/tree_2d_dgsem/elixir_advection_restart.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_restart.jl).

#### Time series
Sometimes it is useful to record the evaluations of state variables over time at
a given set of points. This can be achieved by the [`TimeSeriesCallback`](@ref), which is used,
e.g., in
[`examples/tree_2d_dgsem/elixir_acoustics_gaussian_source.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_acoustics_gaussian_source.jl).
The `TimeSeriesCallback` constructor expects a semidiscretization and a list of points at
which the solution should be recorded in regular time step intervals. After the
last time step, the entire record is stored in an HDF5 file.

For the points, two different input formats are supported: You can either provide
them as a list of tuples, which is handy if you specify them by hand on the
REPL. Alternatively, you can provide them as a two-dimensional array, where the
first dimension is the point number and the second dimension is the
coordinate dimension. This is especially useful when reading them from a file.

For example, to record the primitive variables at the points `(0.0, 0.0)` and
`(-1.0, 0.5)` every five timesteps and storing the collected data in the file
`tseries.h5`, you can create the `TimeSeriesCallback` as
```julia
time_series = TimeSeriesCallback(semi, [(0.0, 0.0), (-1.0, 0.5)];
                                 interval=5,
                                 solution_variables=cons2prim,
                                 filename="tseries.h5")
```
For a full list of possible arguments, please check the documentation for the
[`TimeSeriesCallback`](@ref).
As an alternative to specifying the point coordinates directly in the elixir or
on the REPL, you can read them from a file. For instance, with a text file
`points.dat` with content
```
 0.0 0.0
-1.0 0.5
```
you can create a time series callback with
```julia
using DelimitedFiles: readdlm
time_series = TimeSeriesCallback(semi, readdlm("points.dat"))
```
To plot the individual point data series over time, you can create a
[`PlotData1D`](@ref) from the `TimeSeriesCallback` and a given point ID. For
example, executing
```julia
julia> using Trixi, Plots

julia> trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_acoustics_gaussian_source.jl"))

julia> pd1 = PlotData1D(time_series, 1)

julia> pd2 = PlotData1D(time_series, 2)

julia> plot(pd1["p_prime"]); plot!(pd2["p_prime"], xguide="t")
```
will yield the following plot:

![image](https://user-images.githubusercontent.com/3637659/115822874-9108d900-a405-11eb-9960-4ca3d535e3c6.png)


### Miscellaneous
* The [`AliveCallback`](@ref) prints some information to the screen to show that a simulation is
  still running.
* The [`SummaryCallback`](@ref) prints a human-readable summary of the simulation setup and controls
  the automated performance measurements, including an output of the recorded timers after a simulation.
* The [`VisualizationCallback`](@ref) can be used for in-situ visualization. See
  [Visualizing results during a simulation](@ref).
* The [`TrivialCallback`](@ref) does nothing and can be used to easily disable some callbacks
  via [`trixi_include`](@ref).

### Equation-specific callbacks
Some callbacks provided by Trixi.jl implement specific features for certain equations:
* The [`LBMCollisionCallback`](@ref) implements the Lattice-Boltzmann method (LBM) collision
  operator and should only be used when solving the Lattice-Boltzmann equations. See e.g.
  [`examples/tree_2d_dgsem/elixir_lbm_constant.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_lbm_constant.jl)
* The [`SteadyStateCallback`](@ref) terminates the time integration when the residual steady state
  falls below a certain threshold. This checks the convergence of the potential ``\phi`` for
  hyperbolic diffusion. See e.g. [`examples/tree_2d_dgsem/elixir_hypdiff_nonperiodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_hypdiff_nonperiodic.jl).
* The [`GlmSpeedCallback`](@ref) updates the divergence cleaning wave speed `c_h` for the ideal
  GLM-MHD equations. See e.g. [`examples/tree_2d_dgsem/elixir_mhd_alfven_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_mhd_alfven_wave.jl).

## Usage of step callbacks
Step callbacks are passed to the `solve` method from the ODE solver via the keyword argument
`callback`. If you want to use a single callback `cb`, pass it as `callback=cb`. When using two or
more callbacks, you need to turn them into a `CallbackSet` first by calling
`callbacks = CallbackSet(cb1, cb2)` and passing it as `callback=callbacks`.

!!! note
    There are some restrictions regarding the order of callbacks in a `CallbackSet`.

    The callbacks are called *after* each time step but some callbacks actually belong to the next
    time step. Therefore, the callbacks should be ordered in the following way:
    * Callbacks that belong to the current time step:
        * `SummaryCallback` controls, among other things, timers and should thus be first
        * `SteadyStateCallback` may mark a time step as the last one
        * `AnalysisCallback` may do some checks that mark a time step as the last one
        * `AliveCallback` should be nearby `AnalysisCallback`
        * `SaveSolutionCallback`/`SaveRestartCallback` should save the current solution before it is
          degraded by AMR
        * `VisualizationCallback` should be called before the mesh is adapted
    * Callbacks that belong to the next time step:
        * `AMRCallback`
        * `StepsizeCallback` must be called after `AMRCallback` to accommodate potential changes to
          the mesh
        * `GlmSpeedCallback` must be called after `StepsizeCallback` because the step size affects
          the value of `c_h`
        * `LBMCollisionCallback` is already part of the calculations of the next time step and
          should therefore be called after `StepsizeCallback`


## Stage callbacks
[`PositivityPreservingLimiterZhangShu`](@ref) is a positivity-preserving limiter, used to enforce
physical constraints. An example elixir using this feature can be found at
[`examples/tree_2d_dgsem/elixir_euler_positivity.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_positivity.jl).

## Implementing new callbacks
Since Trixi.jl is compatible with [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl),
both packages share the same callback interface. A detailed description of it can be found in the
OrdinaryDiffEq.jl [documentation](https://diffeq.sciml.ai/latest/).
Step callbacks are just called [callbacks](https://diffeq.sciml.ai/latest/features/callback_functions/).
Stage callbacks are called [`stage_limiter!`](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws)).

An example elixir showing how to implement a new simple stage callback and a new simple step
callback can be found at [`examples/tree_2d_dgsem/elixir_advection_callbacks.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_callbacks.jl).
