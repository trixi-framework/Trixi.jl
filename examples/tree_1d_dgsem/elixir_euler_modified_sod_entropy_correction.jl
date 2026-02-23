using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations1D(1.4)

function initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)
    if x[1] < 0.3
        return prim2cons(SVector(1, 0.75, 1), equations)
    else
        # this version of modified sod uses a 100x density and pressure jump
        return prim2cons(SVector(0.0125, 0.0, 0.01), equations)
    end
end

initial_condition = initial_condition_modified_sod

volume_flux = flux_central
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
indicator_ec = IndicatorEntropyCorrection(equations, basis)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
indicator = IndicatorEntropyCorrectionShockCapturingCombined(indicator_entropy_correction = indicator_ec,
                                                             indicator_shock_capturing = indicator_sc)

volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                                     volume_flux_fv = surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)
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

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

###############################################################################
# run the simulation

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)
sol = solve(ode, SSPRK43();
            abstol = 1e-6, reltol = 1e-4,
            ode_default_options()..., callback = callbacks);
