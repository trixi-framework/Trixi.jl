using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = PengRobinson()
equations = NonIdealCompressibleEulerEquations1D(eos)

# the Peng-Robinson N2 transcritical shock taken from "An entropy-stable hybrid scheme 
# for simulations of transcritical real-fluid flows" by Ma, Ihme (2017).
# <https://doi.org/10.1016/j.jcp.2017.03.022>
function initial_condition_transcritical_shock(x, t,
                                               equations::NonIdealCompressibleEulerEquations1D{<:PengRobinson})
    RealT = eltype(x)
    eos = equations.equation_of_state

    if x[1] < 0
        rho, v1, p = SVector(800, 0, 60e6)
    else
        rho, v1, p = SVector(80, 0, 6e6)
    end

    V = inv(rho)

    # invert for temperature given p, V
    T = eos.Tc
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
        T = max(tol, T - dp / dpdT_V)
        iter += 1
    end
    if iter == 100
        @warn "Solver for temperature(V, p) did not converge"
    end

    return prim2cons(SVector(V, v1, T), equations)
end
initial_condition = initial_condition_transcritical_shock

volume_flux = flux_central_terashima_etal
solver = DGSEM(polydeg = 3, volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
               surface_flux = flux_lax_friedrichs)

coordinates_min = -0.5
coordinates_max = 0.5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 30_000,
                periodicity = false)

boundary_conditions = (x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0e-4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.25)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
