using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 20.0

N_passes = 1
T_end = EdgeLength * N_passes
tspan = (0.0, T_end)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # Evaluate error after full domain traversion
    if t == T_end
        t = 0
    end

    # initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # strength of the vortex
    S = 13.5
    # Radius of vortex
    R = 1.5
    # Free-stream Mach 
    M = 0.4
    # base flow
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)

    cent = inicenter + vel * t # advection of center
    cent = x - cent            # distance to centerpoint
    cent = SVector(cent[2], -cent[1])
    r2 = cent[1]^2 + cent[2]^2

    f = (1 - r2) / (2 * R^2)

    rho = (1 - (S * M / pi)^2 * (gamma - 1) * exp(2 * f) / 8)^(1 / (gamma - 1))

    du = S / (2 * π * R) * exp(f) # vel. perturbation
    vel = vel + du * cent
    v1, v2 = vel

    p = rho^gamma / (gamma * M^2)
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

# Volume flux stabilizes the simulation - in contrast to standard DGSEM with 
# `surface_flux = flux_ranocha` only which crashes.
# To turn this into a convergence test, use a flux with some dissipation, e.g.
# `flux_lax_friedrichs` or `flux_hll`.
solver = DGSEM(polydeg = 2, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

coordinates_min = (-EdgeLength / 2, -EdgeLength / 2)
coordinates_max = (EdgeLength / 2, EdgeLength / 2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 10,
                                     analysis_errors = Symbol[], # Switch off error computation
                                     analysis_integrals = (entropy,),
                                     save_analysis = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

relaxation_solver = Trixi.RelaxationSolverBisection(max_iterations = 20,
                                                    root_tol = eps(Float64),
                                                    gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationCKL43(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0, save_everystep = false, callback = callbacks);
