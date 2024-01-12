
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5 / 3
equations = CompressibleEulerEquations3D(gamma)

"""
    initial_condition_blob(x, t, equations::CompressibleEulerEquations3D)

The blob test case taken from
- Agertz et al. (2006)
  Fundamental differences between SPH and grid methods
  [arXiv: astro-ph/0610051](https://arxiv.org/abs/astro-ph/0610051)
"""
function initial_condition_blob(x, t, equations::CompressibleEulerEquations3D)
    # blob test case, see Agertz et al. https://arxiv.org/pdf/astro-ph/0610051.pdf
    # other reference: https://arxiv.org/pdf/astro-ph/0610051.pdf
    # change discontinuity to tanh
    # typical domain is rectangular, we change it to a square
    # resolution 128^3, 256^3
    # domain size is [-20.0,20.0]^3
    # gamma = 5/3 for this test case
    R = 1.0 # radius of the blob
    # background density
    rho = 1.0
    Chi = 10.0 # density contrast
    # reference time of characteristic growth of KH instability equal to 1.0
    tau_kh = 1.0
    tau_cr = tau_kh / 1.6 # crushing time
    # determine background velocity
    v1 = 2 * R * sqrt(Chi) / tau_cr
    v2 = 0.0
    v3 = 0.0
    Ma0 = 2.7 # background flow Mach number Ma=v/c
    c = v1 / Ma0 # sound speed
    # use perfect gas assumption to compute background pressure via the sound speed c^2 = gamma * pressure/density
    p = c * c * rho / equations.gamma
    # initial center of the blob
    inicenter = [-15, 0, 0]
    x_rel = x - inicenter
    r = sqrt(x_rel[1]^2 + x_rel[2]^2 + x_rel[3]^2)
    # steepness of the tanh transition zone
    slope = 2
    # density blob
    rho = rho +
          (Chi - 1) * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
    # velocity blob is zero
    v1 = v1 - v1 * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_blob

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-20.0, -20.0, -20.0)
coordinates_max = (20.0, 20.0, 20.0)

refinement_patches = ((type = "box", coordinates_min = (-20.0, -10.0, -10.0),
                       coordinates_max = (-10.0, 10.0, 10.0)),
                      (type = "box", coordinates_min = (-20.0, -5.0, -5.0),
                       coordinates_max = (-10.0, 5.0, 5.0)),
                      (type = "box", coordinates_min = (-17.0, -2.0, -2.0),
                       coordinates_max = (-13.0, 2.0, 2.0)),
                      (type = "box", coordinates_min = (-17.0, -2.0, -2.0),
                       coordinates_max = (-13.0, 2.0, 2.0)))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 200,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = IndicatorLöhner(semi,
                                variable = Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 1,
                                      med_level = 0, med_threshold = 0.1, # med_level = current level
                                      max_level = 6, max_threshold = 0.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 3,
                           adapt_initial_condition = false,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-4, 1.0e-4),
                                                     variables = (Trixi.density, pressure))

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
