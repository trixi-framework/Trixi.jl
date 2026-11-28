using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mach_number() = 0.1

# Use Sutherland's law for a temperature-dependent viscosity.
# For details, see e.g.
# Frank M. White: Viscous Fluid Flow, 2nd Edition.
# 1991, McGraw-Hill, ISBN, 0-07-069712-4
# Pages 28 and 29.
# We use the reference temperature and the Sutherland constant of air.
# The reference viscosity is chosen such that the Reynolds number at the
# reference temperature is Re = 1600 as in `elixir_navierstokes_taylor_green_vortex.jl`.
temperature_ref() = 291.15 # [K]
sutherland_constant_air() = 120.0 # [K]
mu_ref() = 6.25e-4 # equivalent to Re = 1600 at the reference temperature

@inline function mu(u_prim_temperature,
                    equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
    RealT = eltype(u_prim_temperature)
    T_ref = convert(RealT, temperature_ref())
    C_air = convert(RealT, sutherland_constant_air())
    _, _, _, T = u_prim_temperature

    return convert(RealT, mu_ref()) * (T_ref + C_air) / (T + C_air) * (T / T_ref)^1.5f0
end

equations = CompressibleEulerEquations2D(1.4)

# The initial condition below is non-dimensionalized with the density, the maximal
# velocity, and the length scale of the vortex. Then, its background pressure is
# p = 1 / (gamma * Ma^2). We choose the specific gas constant R such that the
# corresponding background temperature T = p / (R * rho) is the reference temperature
# of Sutherland's law in Kelvin.
R = 1 / (equations.gamma * mach_number()^2 * temperature_ref())
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number(),
                                                          R = R)

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
    A = 1 # magnitude of speed
    Ms = convert(RealT, mach_number()) # maximum Mach number

    rho = 1
    v1 = A * sin(x[1]) * cos(x[2])
    v2 = -A * cos(x[1]) * sin(x[2])

    # Scaling to get Ms
    p = (A / Ms)^2 * rho / equations.gamma
    # Add terms for the Taylor-Green vortex
    p = p + 0.25f0 * A^2 * rho * (cos(2 * x[1]) + cos(2 * x[2]))

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
sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.Threaded());
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
