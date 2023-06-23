
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the two-layer shallow water equations to test well-balancedness

equations = ShallowWaterTwoLayerEquations1D(gravity_constant=1.0, H0=0.6, rho_upper=0.9, rho_lower=1.0)

"""
    initial_condition_fjordholm_well_balanced(x, t, equations::ShallowWaterTwoLayerEquations1D)

Initial condition to test well balanced with a bottom topography from Fjordholm
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  [DOI: 10.1142/9789814417099_0039](https://doi.org/10.1142/9789814417099_0039)
"""
function initial_condition_fjordholm_well_balanced(x, t, equations::ShallowWaterTwoLayerEquations1D)
    inicenter = 0.5
    x_norm = x[1] - inicenter
    r = abs(x_norm)

    H_lower = 0.5
    H_upper = 0.6
    v1_upper = 0.0
    v1_lower = 0.0
    b  = r <= 0.1 ? 0.2 * (cos(10 * pi * (x[1] - 0.5)) + 1) : 0.0
    return prim2cons(SVector(H_upper, v1_upper, H_lower, v1_lower, b), equations)
  end

initial_condition = initial_condition_fjordholm_well_balanced

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_es_fjordholm_etal, flux_nonconservative_fjordholm_etal),
              volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(lake_at_rest_error,))

stepsize_callback = StepsizeCallback(cfl=1.0)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
