using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations1D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations1D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0)
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)
    # The following code is equivalent to
    # phi = atan(0.0, x_norm)
    # cos_phi = cos(phi)
    # in 1D but faster
    cos_phi = x_norm > 0 ? one(x_norm) : -one(x_norm)

    # Calculate primitive variables
    rho = r > 0.5f0 ? one(RealT) : RealT(1.1691)
    v1 = r > 0.5f0 ? zero(RealT) : RealT(0.1882) * cos_phi
    p = r > 0.5f0 ? RealT(1.0E-3) : RealT(1.245)

    return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_blast_wave

# Note: We do not need to use the shock-capturing methodology here,
# in contrast to the standard `euler_blast_wave.jl` example.
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc)

coordinates_min = (-2.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

stage_limiter! = EntropyBoundedLimiter()

###############################################################################
# run the simulation

sol = solve(ode, SSPRK33(stage_limiter!);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()...,
            callback = callbacks);
