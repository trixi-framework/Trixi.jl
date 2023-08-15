
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations
#
# TODO: TrixiShallowWater: wet/dry example elixir

equations = ShallowWaterEquations2D(gravity_constant=9.81)

"""
    initial_condition_parabolic_bowl(x, t, equations:: ShallowWaterEquations2D)

Well-known initial condition to test the [`hydrostatic_reconstruction_chen_noelle`](@ref) and its
wet-dry mechanics. This test has an analytical solution. The initial condition is defined by the
analytical solution at time t=0. The bottom topography defines a bowl and the water level is given
by an oscillating lake.

The original test and its analytical solution were first presented in
- William C. Thacker (1981)
  Some exact solutions to the nonlinear shallow-water wave equations
  [DOI: 10.1017/S0022112081001882](https://doi.org/10.1017/S0022112081001882).

The particular setup below is taken from Section 6.2 of
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and Timothy Warburton (2018)
  An entropy stable discontinuous Galerkin method for the shallow water equations on
  curvilinear meshes with wet/dry fronts accelerated by GPUs
  [DOI: 10.1016/j.jcp.2018.08.038](https://doi.org/10.1016/j.jcp.2018.08.038).
"""
function initial_condition_parabolic_bowl(x, t, equations:: ShallowWaterEquations2D)
  a = 1.0
  h_0 = 0.1
  sigma = 0.5
  ω = sqrt(2 * equations.gravity * h_0) / a

  v1 = -sigma * ω * sin(ω * t)
  v2 = sigma * ω * cos(ω * t)

  b = h_0 * ((x[1])^2 + (x[2])^2) / a^2

  H = sigma * h_0 / a^2 * (2 * x[1] * cos(ω * t) + 2 * x[2] * sin(ω * t) - sigma) + h_0

  # It is mandatory to shift the water level at dry areas to make sure the water height h
  # stays positive. The system would not be stable for h set to a hard 0 due to division by h in
  # the computation of velocity, e.g., (h v1) / h. Therefore, a small dry state threshold
  # with a default value of 500*eps() ≈ 1e-13 in double precision, is set in the constructor above
  # for the ShallowWaterEquations and added to the initial condition if h = 0.
  # This default value can be changed within the constructor call depending on the simulation setup.
  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_parabolic_bowl


###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_chen_noelle, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassnerShallowWater(equations, basis,
                                                     alpha_max=0.6,
                                                     alpha_min=0.001,
                                                     alpha_smooth=true,
                                                     variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)


###############################################################################

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)

cells_per_dimension = (150, 150)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

stage_limiter! = PositivityPreservingLimiterShallowWater(variables=(Trixi.waterheight,))

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!);
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
