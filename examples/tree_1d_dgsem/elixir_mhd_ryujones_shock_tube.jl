
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 5 / 3
equations = IdealGlmMhdEquations1D(gamma)

"""
    initial_condition_ryujones_shock_tube(x, t, equations::IdealGlmMhdEquations1D)

Ryu and Jones shock tube test case for one dimensional ideal MHD equations. Contains
fast shocks, slow shocks, and rational discontinuities that propagate on either side
of the contact discontinuity. Exercises the scheme to capture all 7 types of waves
present in the one dimensional MHD equations. It is the second test from Section 4 of
- Ryu and Jones (1995)
  Numerical Magnetohydrodynamics in Astrophysics: Algorithm and Tests
  for One-Dimensional Flow
  [DOI: 10.1086/175437](https://doi.org/10.1086/175437)
!!! note
    This paper has a typo in the initial conditions. Their variable `E` should be `p`.
"""
function initial_condition_ryujones_shock_tube(x, t, equations::IdealGlmMhdEquations1D)
    # domain must be set to [0, 1], γ = 5/3, final time = 0.2
    rho = x[1] <= 0.5 ? 1.08 : 1.0
    v1 = x[1] <= 0.5 ? 1.2 : 0.0
    v2 = x[1] <= 0.5 ? 0.01 : 0.0
    v3 = x[1] <= 0.5 ? 0.5 : 0.0
    p = x[1] <= 0.5 ? 0.95 : 1.0
    inv_sqrt4pi = 1.0 / sqrt(4 * pi)
    B1 = 2 * inv_sqrt4pi
    B2 = x[1] <= 0.5 ? 3.6 * inv_sqrt4pi : 4.0 * inv_sqrt4pi
    B3 = B1

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3), equations)
end
initial_condition = initial_condition_ryujones_shock_tube

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_hlle
volume_flux = flux_hindenlang_gassner
basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = Trixi.density)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 7,
                n_cells_max = 10_000,
                periodicity = false)

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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
