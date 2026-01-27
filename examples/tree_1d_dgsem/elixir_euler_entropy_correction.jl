using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations1D(1.4)

function initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)
    if x[1] < 0.3
        return prim2cons(SVector(1, 0.75, 1), equations)
    else
        return prim2cons(SVector(0.125, 0.0, 0.1), equations)
    end
end

initial_condition = initial_condition_modified_sod

volume_flux = flux_central
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
volume_integral = VolumeIntegralEntropyCorrection(equations, basis;
                                                  volume_flux_dg = volume_flux,
                                                  volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 30_000,
                periodicity = false)

boundary_conditions = (x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = BoundaryConditionDirichlet(initial_condition))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.20)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
