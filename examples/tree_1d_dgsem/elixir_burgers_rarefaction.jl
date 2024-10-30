
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

basis = LobattoLegendreBasis(3)
# Use shock capturing techniques to suppress oscillations at discontinuities
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)

volume_flux = flux_ec
surface_flux = flux_lax_friedrichs

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinate_min = 0.0
coordinate_max = 1.0

# Make sure to turn periodicity explicitly off as special boundary conditions are specified
mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000,
                periodicity = false)

# Discontinuous initial condition (Riemann Problem) leading to a rarefaction fan.
function initial_condition_rarefaction(x, t, equation::InviscidBurgersEquation1D)
    scalar = x[1] < 0.5 ? 0.5 : 1.5

    return SVector(scalar)
end

###############################################################################
# Specify non-periodic boundary conditions

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_rarefaction)

function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                    surface_flux_function,
                                    equations::InviscidBurgersEquation1D)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

boundary_conditions = (x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

initial_condition = initial_condition_rarefaction

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 42, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
