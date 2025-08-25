
using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection diffusion equation

advection_velocity = 0.1
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.1
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, # set maximum capacity of tree data structure
                periodicity = true)

function x_trans_periodic(x, domain_length = SVector(2 * pi), center = SVector(0.0))
    x_normalized = x .- center
    x_shifted = x_normalized .% domain_length
    x_offset = ((x_shifted .< -0.5 * domain_length) - (x_shifted .> 0.5 * domain_length)) .*
               domain_length
    return center + x_shifted + x_offset
end

function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x_trans_periodic(x - equation.advection_velocity * t)

    nu = diffusivity()
    c = 0.0
    A = 1.0
    L = 2
    f = 1 / L
    omega = 1.0
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

alive_callback = AliveCallback(analysis_interval = 100)

# Stepsize callback which selects the timestep according to the most restrictive CFL condition.
# For coarser grids, linear stability is governed by the advective CFL condition,
# while for high refinements the flow becomes diffusion-dominated.
stepsize_callback = StepsizeCallback(cfl = 1.6,
                                     cfl_diffusive = 0.4)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
