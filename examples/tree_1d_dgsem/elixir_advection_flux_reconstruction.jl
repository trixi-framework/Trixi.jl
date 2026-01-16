# Example elixir demonstrating the Flux Reconstruction method
# This is a simple 1D advection equation test case

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with Flux Reconstruction surface integral
# The volume integral remains the standard weak form
polydeg = 3
basis = LobattoLegendreBasis(3)

surface_flux = flux_lax_friedrichs
surface_integral = SurfaceIntegralFluxReconstruction(basis, surface_flux = surface_flux)

solver = DGSEM(polydeg = polydeg,
               surface_integral = surface_integral,
               volume_integral = VolumeIntegralStrongForm())

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
