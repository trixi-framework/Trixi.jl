# [Restart simulation](@id restart)

You can continue running an already finished simulation by first
preparing the simulation for the restart and then performing the restart.
Here we suppose that in the first run your simulation stops at time 1.0
and then you want it to run further to time 2.0.

## [Prepare the simulation for a restart](@id restart_preparation)
In you original elixir you need to specify to write out restart files.
Those will later be read for the restart of your simulation.
This is done almost the same way as writing the snapshots using the
[`SaveSolutionCallback`](@ref) callback.
For the restart files it is called [`SaveRestartCallback`](@ref):
```julia
save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)
```
Make this part of your `CallbackSet`.

An example is
[`examples/examples/structured_2d_dgsem/elixir_advection_extended.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/structured_2d_dgsem/elixir_advection_extended.jl).


## [Perform the simulation restart](@id restart_perform)
Since all of the information about the simulation can be obtained from the
last snapshot, the restart can be done with relatively few lines
in an extra elixir file.
However, some might prefer to keep everything in one elixir and
conditionals like `if restart` with a boolean variable `restart` that is user defined.

First we need to define from which file we want to restart, e.g.
```julia
restart_file = "restart_000000021.h5"
restart_filename = joinpath("out", restart_file)
```

Then we load the mesh file:
```julia
mesh = load_mesh(restart_filename)
```

This is then needed for the semidiscretization:
```julia
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
```

We then define a new time span for the simulation that takes as starting
time the one form the snapshot:
```julia
tspan = (load_time(restart_filename), 2.0)
```

We now also take the last `dt`, so that our solver does not need to first find
one to fulfill the CFL condition:
```julia
dt = load_dt(restart_filename)
```

The ODE that we will pass to the solver is now:
```julia
ode = semidiscretize(semi, tspan, restart_filename)
```

You should now define a [`SaveSolutionCallback`](@ref) similar to the
[original simulation](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/structured_2d_dgsem/elixir_advection_extended.jl),
but with `save_initial_solution=false`, otherwise our initial snapshot will be overwritten.
If you are using one file for the original simulation and the restart
you can reuse your [`SaveSolutionCallback`](@ref), but need to set
```julia
save_solution.condition.save_initial_solution = false
```

Before we compute the solution using 
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
we need to set the integrator
and its time step number, e.g.:
```julia
integrator = init(ode, CarpenterKennedy2N54(williamson_condition=false),
                  dt=dt, save_everystep=false, callback=callbacks);
load_timestep!(integrator, restart_filename)
```

Now we can compute the solution:
```julia
sol = solve!(integrator)
```

An example is in [`examples/structured_2d_dgsem/elixir_advection_restart.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/structured_2d_dgsem/elixir_advection_restart.jl).
