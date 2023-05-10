
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
equations = IdealMhdMultiIonEquations2D(gammas = (1.4, 1.4),
                                        charge_to_mass = (1.0, 2.0))

"""
    initial_condition_rotor(x, t, equations::IdealMhdMultiIonEquations2D)

The classical MHD rotor test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_rotor(x, t, equations::IdealMhdMultiIonEquations2D)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ = 1.4
  B1 = 5.0/sqrt(4.0*pi)
  B2 = 0.0
  B3 = 0.0
  
  # first species
  dx = x[1] - 0.25
  dy = x[2] - 0.5
  r = sqrt(dx^2 + dy^2)
  f = (0.115 - r)/0.015
  if r <= 0.1
    rho = 10.0
    v1 = -20.0 * dy
    v2 = 20.0 * dx
  elseif r >= 0.115
    if x[1] > 0.75
      rho = 0.49 * (tanh(50 * (x[1] - 1.0)) + 1) + 0.02
    elseif x[1] > 0.25
      rho = 0.49 * (-tanh(50 * (x[1] - 0.5)) + 1) + 0.02
    else
      rho = 0.49 * (tanh(50 * (x[1])) + 1) + 0.02
    end
    v1 = 0.0
    v2 = 0.0
  else
    rho = 1.0 + 9.0*f
    v1 = -20.0*f*dy
    v2 = 20.0*f*dx
  end
  v3 = 0.0
  p = 1.0

  #second species
  dx = x[1] - 0.75
  dy = x[2] - 0.5
  r = sqrt(dx^2 + dy^2)
  f = (0.115 - r)/0.015
  if r <= 0.1
    rho2 = 10.0
    v12 = -20.0 * dy
    v22 = 20.0 * dx
  elseif r >= 0.115
    if x[1] < 0.25
      rho2 = 0.49 * (-tanh(50 * (x[1])) + 1) + 0.02
    elseif x[1] < 0.75
      rho2 = 0.49 * (tanh(50 * (x[1] - 0.5)) + 1) + 0.02
    else
      rho2 = 0.49 * (-tanh(50 * (x[1] - 1.0)) + 1) + 0.02
    end
    v12 = 0.0
    v22 = 0.0
  else
    rho2 = 1.0 + 9.0 * f
    v12 = -20.0 * f * dy
    v22 = 20.0 * f * dx
  end
  v3 = 0.0
  p = 1.0
  
  return prim2cons(SVector(B1, B2, B3, rho, v1, v2, v3, p, rho2, v12, v22, v3, p), equations)
end
initial_condition = initial_condition_rotor

volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_standard)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)
amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      max_level =7, max_threshold=0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=6,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

save_restart = SaveRestartCallback(interval=100,
                           save_final_restart=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
			                  amr_callback,
                        save_solution,
                        save_restart,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
