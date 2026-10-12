using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)

Spherical blast wave into a near-vacuum ambient state. The low outer density and
pressure make cell averages of the internal energy go negative without limiting,
allowing the Liu-Zhang limiter to run to find an admissible solution.
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)
    RealT = eltype(x)
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    r0 = 0.4f0

    if r < r0
        rho = one(RealT)
        p = one(RealT)
    elseif r ≈ r0
        rho = RealT(0.5) * (1 + RealT(1.0e-3))
        p = RealT(0.5) * (1 + RealT(1.0e-10))
    else
        rho = RealT(1.0e-3)
        p = RealT(1.0e-10)
    end

    return prim2cons(SVector(rho, 0, 0, 0, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
solver = DGSEM(basis, surface_flux, VolumeIntegralWeakForm())

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.4)

###############################################################################
# run the simulation

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback, stepsize_callback)

# Liu-Zhang limiting thresholds and callback setup
thresholds = (1.0e-12, 1.0e-12)
local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = thresholds,
                                                     variables = (Trixi.density,
                                                                  energy_internal))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

ode_solver = CarpenterKennedy2N54(; stage_limiter! = global_limiter!,
                                  step_limiter! = global_limiter!,
                                  williamson_condition = false, thread = Trixi.Threaded())

sol = solve(ode, ode_solver;
            dt = 1, # overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
