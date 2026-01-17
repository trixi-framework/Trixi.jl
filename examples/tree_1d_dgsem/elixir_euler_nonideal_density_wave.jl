using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = VanDerWaals(; a = 10, b = 1e-2, gamma = 1.4, R = 287)
equations = NonIdealCompressibleEulerEquations1D(eos)

# the default amplitude and frequency k are chosen to be consistent with 
# initial_condition_density_wave for CompressibleEulerEquations1D
function initial_condition_density_wave(x, t,
                                        equations::NonIdealCompressibleEulerEquations1D;
                                        amplitude = 0.98, k = 2)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    rho = 1 + convert(RealT, amplitude) * sinpi(k * (x[1] - v1 * t))
    p = 20

    V = inv(rho)

    # invert for temperature given p, V
    T = 1
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
        T = T - dp / dpdT_V
        iter += 1
    end
    if iter == 100
        println("Warning: solver for temperature(V, p) did not converge")
    end

    return prim2cons(SVector(V, v1, T), equations)
end

initial_condition = initial_condition_density_wave

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
