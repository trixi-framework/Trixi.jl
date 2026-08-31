using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72

# Use Sutherland's law for a temperature-dependent viscosity.
# For details, see e.g.
# Frank M. White: Viscous Fluid Flow, 2nd Edition.
# 1991, McGraw-Hill, ISBN, 0-07-069712-4
# Pages 28 and 29.
@inline function mu(u_prim_temperature, equations)
    RealT = eltype(u_prim_temperature)
    T_ref = convert(RealT, 291.15) # [K]
    _, _, _, T = u_prim_temperature

    C_air = 120 # [K]
    mu_ref_air = convert(RealT, 1.827e-5) # [Pa * s] = [kg/(m * s)]

    return mu_ref_air * (T_ref + C_air) / (T + C_air) * (T / T_ref)^1.5f0
end

equations = CompressibleEulerEquations2D(1.4)

R_specific_air = 287.052874 # [J/(kg * K)]
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number(),
                                                          R = R_specific_air)

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
    RealT = eltype(x)
    A = 1 # magnitude of speed [m/s]
    Ms = convert(RealT, 0.1) # maximum Mach number

    rho = 1 # [kg/m^3]
    v1 = A * sin(x[1]) * cos(x[2]) # [m/s]
    v2 = -A * cos(x[1]) * sin(x[2]) # [m/s]

    # Scaling to get Ms
    p = (A / Ms)^2 * rho / equations.gamma # [Pa] = [kg/(m * s^2)]
    # Add terms for the Taylor-Green vortex
    p = p + 0.25f0 * A^2 * rho * (cos(2 * x[1]) + cos(2 * x[2])) # [Pa] = [kg/(m * s^2)]

    println("p: ", p)

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
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
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

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
