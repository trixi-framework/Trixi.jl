using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

basis = LobattoLegendreBasis(3)
# Use shock capturing techniques to suppress oscillations at discontinuities
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)

volume_flux = flux_ec
surface_flux = flux_lax_friedrichs

volume_integral_hg = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(flux_ec)
volume_integral_weakform = VolumeIntegralWeakForm() # unstable

indicator = IndicatorEntropyChange(maximum_entropy_increase = 0.0)

# Adaptive volume integral using the entropy change indicator to perform the 
# stabilized/EC volume integral when needed and keeping the weak form if it is more diffusive.
volume_integral = VolumeIntegralAdaptive(volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff,
                                         indicator = indicator)

solver = DGSEM(basis, surface_flux, volume_integral_fluxdiff)

coordinate_min = 0.0
coordinate_max = 1.0

# Make sure to turn periodicity explicitly off as special boundary conditions are specified
mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

function initial_condition_shock(x, t, equation::InviscidBurgersEquation1D)
    scalar = sinpi(2 * x[1]) + 0.5

    return SVector(scalar)
end


initial_condition = initial_condition_shock

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

t_discontinuous = 1/(2 * pi) # time when shock forms
tspan = (0.0, t_discontinuous)

t_end = 0.35 # 0.35 (there WF has already crasehed)
# t_end = 10.0 # test long time stability
tspan = (0.0, t_end)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK54(); dt = 1.0,
            ode_default_options()..., callback = callbacks);
