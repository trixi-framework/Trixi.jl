
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the two-layer shallow water equations

equations = TwoLayerShallowWaterEquations1D(gravity_constant=1.0,H0=0.6,rho1=1.0,rho2=0.9)

# Initial Conditions from Fjordholm testcase
function initial_condition_eec_test(x, t, equations::TwoLayerShallowWaterEquations1D)
    add_perturbation = false
    inicenter = 0.5
    x_norm = x[1] - inicenter
    r = abs(x_norm)
  
    # Add perturbation to h2
    if add_perturbation == true
      h2 = 0.38<=x[1]<=0.42 ? 0.15 : 0.1
    else 
      h2 = 0.1
    end
  
    v1 = 0.0
    v2 = 0.0
    H1 = 0.5
    H2 = H1 + h2
    b  = r <= 0.1 ? 0.2 * (cos(10*π*(x[1] - 0.5)) + 1) : 0.0
    return prim2cons(SVector(H1, H2, v1, v2, b), equations)
  end

initial_condition = initial_condition_eec_test

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
              volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10_000,
                periodicity=true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,save_analysis=true,extra_analysis_integrals=(lake_at_rest_error,))

stepsize_callback = StepsizeCallback(cfl=1.0)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

