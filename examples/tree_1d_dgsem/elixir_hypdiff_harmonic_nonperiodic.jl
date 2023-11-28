
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations1D(nu = 1.25)

"""
    initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations1D)

A non-priodic harmonic function used in combination with
[`source_terms_poisson_nonperiodic`](@ref) and [`boundary_condition_poisson_nonperiodic`](@ref).

!!! note
    The only harmonic functions in 1D have the form phi(x) = A + Bx
"""
function initial_condition_harmonic_nonperiodic(x, t,
                                                equations::HyperbolicDiffusionEquations1D)
    # elliptic equation: -νΔϕ = f
    if t == 0.0
        phi = 5.0
        q1 = 0.0
    else
        A = 3
        B = exp(1)
        phi = A + B * x[1]
        q1 = B
    end
    return SVector(phi, q1)
end
initial_condition = initial_condition_harmonic_nonperiodic

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 30_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_terms_harmonic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 30.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.75)

callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.HypDiffN3Erk3Sstar52(),
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
