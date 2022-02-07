#src # Non-periodic boundary conditions

# Besides the typical periodic boundary condition [`boundary_condition_periodic`](@ref) which is the
# default boundary condition in [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) there are
# some non-periodic BC as well.


# # Dirichlet boundary condition
# We first have a look at the Dirichlet boundary condition [`BoundaryConditionDirichlet`](@ref).
# ```julia
# BoundaryConditionDirichlet(boundary_value_function)
# ```
# In Trixi, that creates a Dirichlet boundary condition that uses the function `boundary_value_function`
# to specify the values at the boundary. It can be used to create a boundary condition that specifies
# exact boundary values by passing the exact solution of the equation.

# The passed boundary value function will be called with the same arguments as an initial condition
# function is called, i.e., as
# ```julia
# boundary_value_function(x, t, equations)
# ```
# where `x` specifies the coordinates, `t` is the current time, and `equation` is the corresponding
# system of equations.


# We want to give a short example for a simulation with Dirichlet BC.

# Consider the one-dimensional linear advection equation with domain $\Omega=[0, 2]$ and a constant
# zero initial condition.
using OrdinaryDiffEq, Trixi

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

initial_condition_zero(x, t, equation::LinearScalarAdvectionEquation1D) = SVector(0.0)
initial_condition = initial_condition_zero

using Plots
plot(x -> sum(initial_condition(x, 0.0, equations)), label="initial condition", ylim=(-1.5, 1.5))

# Using an advection velocity of `1.0` and the (local) Lax-Friedrichs/Rusanov flux
# [`FluxLaxFriedrichs`](@ref) we create an inflow boundary on the left and an outflow boundary
# on the right. To define the inflow values, we initialize a `boundary_value_function`.
function boundary_condition_sine_sector(x, t, equation::LinearScalarAdvectionEquation1D)
    if 1.0 <= t <= 3.0
        scalar = sin(2 * pi * sum(t - 1.0))
    else
        scalar = 0.0
    end
    return SVector(scalar)
end
boundary_condition = boundary_condition_sine_sector

# We set the BC in negativ and positiv x direction
boundary_conditions = (x_neg=BoundaryConditionDirichlet(boundary_condition),
                       x_pos=BoundaryConditionDirichlet(boundary_condition))
#-
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (0.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=false)


semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition,
                                    solver,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 6.0)
ode = semidiscretize(semi, tspan)

analysis_callback = AnalysisCallback(semi, interval=100,)

stepsize_callback = StepsizeCallback(cfl=1.6)

callbacks = CallbackSet(analysis_callback,
                        stepsize_callback);

# Nodes for the visualization
visnodes = range(tspan[1], tspan[2], 300)

# Run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, saveat=visnodes, callback=callbacks);

using Plots
@gif for step in 1:length(sol.u)
    plot(sol.u[step], semi, ylim=(-1.5, 1.5), legend=true, label="approximation", title="time t=$(round(sol.t[step], digits=5))")
    scatter!([0.0], [sum(boundary_condition(SVector(0.0), sol.t[step], equations))], label="boundary condition")
end
#src # gif(anim, "out/anim.gif", fps = 10)

# As mentioned before, using the `flux_lax_friedrichs` and an advection velocity of `1` for the
# scalar advection equation yields to an inflow boundary on the left and an outflow on the right.
# This can be observed nicely in this animation.



# # Slip wall boundary condition
# Moreover, there are other boundary conditions in Trixi. For instance, you can use the slip wall boundary
# condition. It is defined for various equations and different mesh types. Each combination requires
# its own formulation.

# For example, explanations for the `CompressibleEulerEquations2D` can be found [here](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.boundary_condition_slip_wall-Tuple{Any,%20AbstractVector{T}%20where%20T,%20Any,%20Any,%20Any,%20CompressibleEulerEquations2D}),
# and for the `ShallowWaterEquations2D` [here](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.boundary_condition_slip_wall-Tuple{Any,%20AbstractVector{T}%20where%20T,%20Any,%20Any,%20Any,%20ShallowWaterEquations2D}).

# TODO: Add example for boundary_condition_slip_wall? Which one?
