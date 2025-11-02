using Trixi

"""
    initial_condition_sedov_positivity(x, t, equations::CompressibleEulerEquations2D)
A version of the Sedov blast based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_sedov_positivity(x, t, equations::CompressibleEulerEquations1D)
    # Set up "polar" coordinates
    inicenter = SVector(0.0)
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)

    # Ambient values
    rho_0 = 1
    p_0 = 1.0e-5

    sigma_rho = 0.25
    sigma_p = 0.15

    rho = rho_0 + 1 / (4 * pi * sigma_rho^2) * exp(-0.5 * r^2 / sigma_rho^2)
    v = 0.0
    p = p_0 + (equations.gamma - 1) / (4 * pi * sigma_p^2) * exp(-0.5 * r^2 / sigma_p^2)

    return prim2cons(SVector(rho, v, p), equations)
end
initial_condition = initial_condition_sedov_positivity

equations = CompressibleEulerEquations1D(1.4)

surface_flux = flux_hllc
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3) #  7
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.5,)
coordinates_max = (1.5,)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5 # weak blast
cfl = 0.25 # medium blast
stepsize_callback = StepsizeCallback(cfl = cfl) 

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

limiter! = PositivityPreservingLimiterRuedaRamirezGassner(semi;
                                                          beta = 0.1, root_tol = 1e-9)

###############################################################################
# run the simulation

stage_callbacks = (limiter!,)
stage_callbacks = ()

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
