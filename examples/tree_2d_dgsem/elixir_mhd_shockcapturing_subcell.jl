
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations2D)

An MHD blast wave modified from:
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
This setup needs a positivity limiter for the density.
"""
function initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [-0.5, 0.5] x [-0.5, 0.5], γ = 1.4
    RealT = eltype(x)
    r = sqrt(x[1]^2 + x[2]^2)

    pmax = convert(RealT, 10)
    pmin = convert(RealT, 0.01)
    rhomax = one(RealT)
    rhomin = convert(RealT, 0.01)
    if r <= RealT(0.09)
        p = pmax
        rho = rhomax
    elseif r >= RealT(0.1)
        p = pmin
        rho = rhomin
    else
        p = pmin + (convert(RealT, 0.1) - r) * (pmax - pmin) / convert(RealT, 0.01)
        rho = rhomin + (convert(RealT, 0.1) - r) * (rhomax - rhomin) / convert(RealT, 0.01)
    end
    v1 = 0
    v2 = 0
    v3 = 0
    B1 = 1 / sqrt(4 * convert(RealT, pi))
    B2 = 0
    B3 = 0
    psi = 0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_blast_wave

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell_local_symmetric)
volume_flux = (flux_derigs_etal, flux_nonconservative_powell_local_symmetric)
basis = LobattoLegendreBasis(3)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                positivity_correction_factor = 0.1)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = (0.5, 0.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 0.4
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation
stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
