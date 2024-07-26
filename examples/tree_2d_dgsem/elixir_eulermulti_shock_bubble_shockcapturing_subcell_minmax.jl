using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations

# 1) Dry Air  2) Helium + 28% Air
equations = CompressibleEulerMulticomponentEquations2D(gammas = (1.4, 1.648),
                                                       gas_constants = (0.287, 1.578))

"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{5, 2})

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t,
                                        equations::CompressibleEulerMulticomponentEquations2D{5,
                                                                                              2})
    # bubble test case, see Gouasmi et al. https://arxiv.org/pdf/1904.00972
    # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
    # typical domain is rectangular, we change it to a square, as Trixi can only do squares
    @unpack gas_constants = equations

    # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
    delta = 0.03

    # Region I
    rho1_1 = delta
    rho2_1 = 1.225 * gas_constants[1] / gas_constants[2] - delta
    v1_1 = zero(delta)
    v2_1 = zero(delta)
    p_1 = 101325

    # Region II
    rho1_2 = 1.225 - delta
    rho2_2 = delta
    v1_2 = zero(delta)
    v2_2 = zero(delta)
    p_2 = 101325

    # Region III
    rho1_3 = 1.6861 - delta
    rho2_3 = delta
    v1_3 = -113.5243
    v2_3 = zero(delta)
    p_3 = 159060

    # Set up Region I & II:
    inicenter = SVector(zero(delta), zero(delta))
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    if (x[1] > 0.50)
        # Set up Region III
        rho1 = rho1_3
        rho2 = rho2_3
        v1 = v1_3
        v2 = v2_3
        p = p_3
    elseif (r < 0.25)
        # Set up Region I
        rho1 = rho1_1
        rho2 = rho2_1
        v1 = v1_1
        v2 = v2_1
        p = p_1
    else
        # Set up Region II
        rho1 = rho1_2
        rho2 = rho2_2
        v1 = v1_2
        v2 = v2_2
        p = p_2
    end

    return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
end
initial_condition = initial_condition_shock_bubble

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_twosided_variables_cons = ["rho" * string(i)
                                                                 for i in eachcomponent(equations)])
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.25, -2.225)
coordinates_max = (2.20, 2.225)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 1_000_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 300
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (Trixi.density,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 600,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.4)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback(save_errors = false))

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
