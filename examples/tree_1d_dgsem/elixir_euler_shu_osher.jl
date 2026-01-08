using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqCore: PIDController
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

# Shu-Osher initial condition for 1D compressible Euler equations
# Example 8 from Shu, Osher (1989).
# https://doi.org/10.1016/0021-9991(89)90222-2
function initial_condition_shu_osher(x, t, equations::CompressibleEulerEquations1D)
    x0 = -4

    rho_left = 27 / 7
    v_left = 4 * sqrt(35) / 9
    p_left = 31 / 3

    v_right = 0.0
    p_right = 1.0

    rho = ifelse(x[1] > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
    v = ifelse(x[1] > x0, v_right, v_left)
    p = ifelse(x[1] > x0, p_right, p_left)

    return prim2cons(SVector(rho, v, p), equations)
end

initial_condition = initial_condition_shu_osher

surface_flux = flux_hllc
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator_variable = density_pressure
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = shock_indicator_variable)
volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                  volume_flux_dg = volume_flux,
                                                  volume_flux_fv = surface_flux,
                                                  slope_limiter = monotonized_central)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-5.0,)
coordinates_max = (5.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000,
                periodicity = false)

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = boundary_condition_default(mesh, boundary_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

# Solve ODE with optimized timestep controller from https://doi.org/10.1007/s42967-021-00159-w
sol = solve(ode, SSPRK43();
            controller = PIDController(0.55, -0.27, 0.05),
            abstol = 1e-4, reltol = 1e-4,
            callback = callbacks, ode_default_options()...)
