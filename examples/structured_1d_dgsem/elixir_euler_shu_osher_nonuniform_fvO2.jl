using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

# Shu-Osher initial condition for 1D compressible Euler equations
# Example 8 from Shu, Osher (1989).
# [https://doi.org/10.1016/0021-9991(89)90222-2](https://doi.org/10.1016/0021-9991(89)90222-2)
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
polydeg = 3 # governs in this case only the number of FV subcells per DG cell
basis = LobattoLegendreBasis(polydeg)
volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = surface_flux,
                                                      reconstruction_mode = reconstruction_O2_inner,
                                                      slope_limiter = monotonized_central)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

function refined_mapping(xi)
    fine_fraction = 0.8
    a_fine = 4.0
    x_max = 5.0

    # Compute the value at the boundary for continuity
    x_boundary = a_fine * fine_fraction
    # Slope for the outer region
    a_outer = (x_max - x_boundary) / (1 - fine_fraction)
    if abs(xi) <= fine_fraction
        x = a_fine_region * xi
    else
        x = x_boundary + a_outer * (abs(xi) - fine_fraction)
        x *= sign(xi)
    end
    return x
end

cells_per_dimension = (128,)
mesh = StructuredMesh(cells_per_dimension, refined_mapping, periodicity = false)

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

sol = solve(ode, SSPRK43();
            abstol = 1e-4, reltol = 1e-4,
            callback = callbacks, ode_default_options()...)
