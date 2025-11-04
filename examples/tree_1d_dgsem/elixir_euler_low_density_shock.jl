using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(5/3)

"""
    initial_condition_near_vacuum_shock(x, t, equations::CompressibleEulerEquations1D)

Leblanc shock tube problem inspired initial condition.
Uses much lower pressure but much higher pressure on the right side.
"""
function initial_condition_near_vacuum_shock(x, t, equations::CompressibleEulerEquations1D)
    rho = x[1] <= 3 ? 1 : 1e-5
    rho_v1 = 0
    rho_e = x[1] <= 3 ? 1e-1 : 1e-4

    return SVector(rho, rho_v1, rho_e)
end
initial_condition = initial_condition_near_vacuum_shock

###############################################################################
# Specify non-periodic boundary conditions

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition)

function boundary_condition_outflow(u_inner, orientation, direction, x, t,
                                    surface_flux_function,
                                    equations::CompressibleEulerEquations1D)
    # Calculate the boundary flux entirely from the internal solution state
    return flux(u_inner, orientation, equations)
end

boundary_conditions = (x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

surface_flux = flux_hllc
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(4)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0,)
coordinates_max = (9.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi,
                                variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 4,
                                      med_level = 0, med_threshold = 0.1,
                                      max_level = 6, max_threshold = 0.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 2,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

# Positivity-preserving limiter setup
# - `alpha_max` is increased above the value used in the volume integral 
#               to allow room for positivity limiting.
# - `beta_rho` set to 0.7 to provoke density correction
# - `root_tol` can be set to this relatively high value while still ensuring positivity
# - `use_density_init` set to false to use zero initial guess for pressure correction
limiter! = PositivityPreservingLimiterRuedaRamirezGassner(semi;
                                                          alpha_max = 0.9,
                                                          beta_rho = 0.7,
                                                          root_tol = 1e-8,
                                                          use_density_init = false)

stage_callbacks = (limiter!,)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
