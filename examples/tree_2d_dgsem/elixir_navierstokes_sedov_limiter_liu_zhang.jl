using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu() = 1.0e-2

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A slight modification of the Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.5f0
    E = 1
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * convert(RealT, pi) * r0^2)

    p0_outer = convert(RealT, 1.0e-5)

    # Calculate primitive variables
    rho = 1
    v1 = 0
    v2 = 0
    p = r > r0 ? p0_outer : p0_inner
    if r ≈ r0
        p = 0.5f0 * (p0_inner + p0_outer)
    end

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
indicator_ec = IndicatorEntropyCorrection(equations, basis)
volume_integral_default = VolumeIntegralWeakForm()
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)

dg = DGSEM(basis, surface_flux, volume_integral)

# This maps the domain [-1, 1]^2 to [-2, 2]^2 while also
# introducing a curved warping to interior nodes.
function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return 2 * SVector(x, y)
end

trees_per_dimension = (2, 2)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, 
                 initial_refinement_level = 4,
                 mapping = mapping,
                 periodicity = true)


semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

###############################################################################
# run the simulation

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback)

local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-8, 1e-8),
                                                     variables = (Trixi.density,
                                                                  energy_internal))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

ode_solver = RDPK3SpFSAL35(; stage_limiter! = global_limiter!,
                             step_limiter! = global_limiter!)

sol = solve(ode, ode_solver;
            adaptive = true, dt = 1e-7, abstol = 1e-5, reltol = 1e-5,
            ode_default_options()..., callback = callbacks);
