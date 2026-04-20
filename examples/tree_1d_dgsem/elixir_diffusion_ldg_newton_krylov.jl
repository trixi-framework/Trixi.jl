using Trixi

using OrdinaryDiffEqSDIRK
using LinearSolve # For Jacobian-free Newton-Krylov (GMRES) solver
using ADTypes # For automatic differentiation via finite differences

###############################################################################
# semidiscretization of the pure diffusion equation

diffusivity() = 0.5
equations = LinearDiffusionEquation1D(diffusivity())

solver = DGSEM(polydeg = 3)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, periodicity = true)

function initial_condition_pure_diffusion_1d_convergence_test(x, t,
                                                              equation)
    nu = diffusivity()
    c = 0
    A = 1
    omega = 1
    scalar = c + A * sin(omega * sum(x)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_pure_diffusion_1d_convergence_test

solver_parabolic = ParabolicFormulationLocalDG()
semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 10)
alive_callback = AliveCallback(alive_interval = 1)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

# Tolerances for GMRES residual, see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
atol_lin_solve = 1e-6
rtol_lin_solve = 1e-5

# Jacobian-free Newton-Krylov (GMRES) solver
linsolve = KrylovJL_GMRES(atol = atol_lin_solve, rtol = rtol_lin_solve)

# Use (diagonally) implicit Runge-Kutta, see
# https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Using-Jacobian-Free-Newton-Krylov
ode_alg = KenCarp47(autodiff = AutoFiniteDiff(), linsolve = linsolve)

atol_ode_solve = 1e-5
rtol_ode_solve = 1e-4
sol = solve(ode, ode_alg;
            abstol = atol_ode_solve, reltol = rtol_ode_solve,
            ode_default_options()..., callback = callbacks);
