using OrdinaryDiffEqLowStorageRK
# SummationByParts.jl provides the nodes of the SBP operators on tetrahedra. Loading it
# enables the corresponding package extension of StartUpDG.jl.
using SummationByParts: SummationByParts
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_weak_blast_wave
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

# Shock capturing on tetrahedra with SBP nodes blends the flux differencing volume integral
# with the algebraic low-order method on the SBP nodes.
polydeg = 3
basis = DGMultiBasis(Tet(), polydeg, approximation_type = SBP())

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

cells_per_dimension = (4, 4, 4)
mesh = DGMultiMesh(dg, cells_per_dimension;
                   coordinates_min = (-2.0, -2.0, -2.0),
                   coordinates_max = (2.0, 2.0, 2.0),
                   periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

# The SBP operators on tetrahedra require a smaller CFL number than, e.g., polynomial
# approximations of the same degree, since some of their quadrature weights are small.
stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
