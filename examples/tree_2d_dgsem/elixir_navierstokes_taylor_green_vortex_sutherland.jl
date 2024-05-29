
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72

# Use Sutherland's law for a temperature-dependent viscosity.
# For details, see e.g. 
# Frank M. White: Viscous Fluid Flow, 2nd Edition.
# 1991, McGraw-Hill, ISBN, 0-07-069712-4
# Pages 28 and 29.
@inline function mu(u, equations)
    T_ref = 291.15

    R_specific_air = 287.052874
    T = R_specific_air * Trixi.temperature(u, equations)

    C_air = 120.0
    mu_ref_air = 1.827e-5

    return mu_ref_air * (T_ref + C_air) / (T + C_air) * (T / T_ref)^1.5
end

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical viscous Taylor-Green vortex in 2D.
This forms the basis behind the 3D case found for instance in
  - Jonathan R. Bull and Antony Jameson
  Simulation of the Compressible Taylor Green Vortex using High-Order Flux Reconstruction Schemes
  [DOI: 10.2514/6.2014-3210](https://doi.org/10.2514/6.2014-3210)
"""
function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations2D)
    A = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2])
    v2 = -A * cos(x[1]) * sin(x[2])
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p + 1.0 / 4.0 * A^2 * rho * (cos(2 * x[1]) + cos(2 * x[2]))

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0) .* pi
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-9
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
