using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations for large eddy
# simulations (LES) with the Smagorinsky subgrid-scale model

prandtl_number() = 0.72
mu = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations3D(1.4)
# The resolution of this setup is too coarse for a direct numerical simulation of the
# Taylor-Green vortex at Re = 1600
subgrid_model = Smagorinsky(C_s = 0.17, Prandtl_turbulent = 0.9)
equations_parabolic = CompressibleLES3D(equations, mu = mu, Prandtl = prandtl_number(),
                                        subgrid_model = subgrid_model)

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)

The classical inviscid Taylor-Green vortex.
"""
function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations3D)
    RealT = eltype(x)
    A = 1 # magnitude of speed
    Ms = convert(RealT, 0.1) # maximum Mach number

    rho = one(RealT)
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = zero(RealT)
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p +
        A^2 * rho / 16 *
        (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
         cos(2 * x[2]) * cos(2 * x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hll,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi

trees_per_dimension = (2, 2, 2)

mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (true, true, true), initial_refinement_level = 2)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 50
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal,
                                                                 enstrophy))
save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback, save_solution)

###############################################################################
# run the simulation

ode_alg = RDPK3SpFSAL49(thread = Trixi.Threaded())
time_int_tol = 1e-8
# A `Rational` initial time step `dt` keeps the real type of `semi` for the time
sol = solve(ode, ode_alg;
            # not necessary, added for overwriting in tests
            adaptive = true, dt = 1 // 256,
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
