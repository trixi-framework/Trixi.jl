using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

# Specify the initial condition as a discontinuous initial condition (see docstring of 
# `DiscontinuousFunction` for more information) which comes with a specialized 
# initialization routine suited for the Riemann problems.
# In short, if a discontinuity is right at an interface, the boundary nodes (which are at the same location)
# on that interface will be initialized with the left and right state of the discontinuity, i.e., 
#                         { u_1, if element = left element and x_{element}^{(n)} = x_jump
# u(x_jump, t, element) = {
#                         { u_2, if element = right element and x_{element}^{(1)} = x_jump
# This is realized by shifting the outer DG nodes inwards, i.e., on reference element
# the outer nodes are at `[-1, 1]` are shifted to `[-1 + ε, 1 - ε]` with machine precision `ε`.
struct InitialConditionMediumSedovBlast <: DiscontinuousFunction end

"""
    (initial_condition_medium_sedov_blast_wave::InitialConditionMediumSedovBlast)(x, t, 
                                                                                  equations::CompressibleEulerEquations3D)

The Sedov blast wave setup based on Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114000000000000000
with smaller strength of the initial discontinuity.
"""
function (initial_condition_medium_sedov_blast_wave::InitialConditionMediumSedovBlast)(x, t,
                                                                                       equations::CompressibleEulerEquations3D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    # Setup based on https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114000000000000000
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    E = 1.0
    p0_inner = 3 * (equations.gamma - 1) * E / (4 * pi * r0^2)
    p0_outer = 1.0e-3

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
# Note calling the constructor of the struct: `InitialConditionMediumSedovBlast()` instead of 
# `initial_condition_medium_sedov_blast_wave` !
initial_condition = InitialConditionMediumSedovBlast()

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 5
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (4, 4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 4, initial_refinement_level = 0,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
