using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = PengRobinson()
equations = NonIdealCompressibleEulerEquations1D(eos)

# the smooth Peng-Robinson N2 transcritical wave taken from "An entropy-stable hybrid 
# scheme for simulations of transcritical real-fluid flows" by Ma, Ihme (2017). In this 
# context, the wave is "transcritical" because the solution involves both subcritical 
# and supercritical density and temperature values. 
# 
# <https://doi.org/10.1016/j.jcp.2017.03.022>
function initial_condition_transcritical_wave(x, t,
                                              equations::NonIdealCompressibleEulerEquations1D{<:PengRobinson})
    RealT = eltype(x)
    eos = equations.equation_of_state

    rho_min, rho_max = 56.9, 793.1
    v1 = 100
    rho = 0.5f0 * (rho_min + rho_max) +
          0.5f0 * (rho_max - rho_min) * sin(2 * pi * (x[1] - v1 * t))
    p = 5e6

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
initial_condition = initial_condition_transcritical_wave

volume_integral = VolumeIntegralFluxDifferencing(flux_terashima_etal)
solver = DGSEM(polydeg = 3, volume_integral = volume_integral,
               surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                                                   boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
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
