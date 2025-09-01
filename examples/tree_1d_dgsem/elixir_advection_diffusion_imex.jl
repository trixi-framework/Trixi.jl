using Trixi
using OrdinaryDiffEqBDF # BDF subpackage exports IMEX methods
using LinearSolve # For Jacobian-free Newton-Krylov (GMRES) solver
using ADTypes # For automatic differentiation via finite differences

###############################################################################

advection_velocity = 0.5
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.1
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

function x_trans_periodic(x, domain_length = SVector(oftype(x[1], 2 * pi)),
                          center = SVector(oftype(x[1], 0)))
    x_normalized = x .- center
    x_shifted = x_normalized .% domain_length
    x_offset = ((x_shifted .< -0.5f0 * domain_length) -
                (x_shifted .> 0.5f0 * domain_length)) .*
               domain_length
    return center + x_shifted + x_offset
end

function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation1D)
    x_trans = x_trans_periodic(x - equation.advection_velocity * t)

    nu = diffusivity()
    c = 0
    A = 1
    omega = 1
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

solver_parabolic = ViscousFormulationLocalDG()
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver; solver_parabolic)

###############################################################################

tspan = (0.0, 0.5 * pi / advection_velocity)

# For hyperbolic-parabolic problems, this results in a SciML SplitODEProblem, see e.g.
# https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/#SciMLBase.SplitODEProblem
# These exactly fit IMEX (implicit-explicit) integrators
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
alive_callback = AliveCallback(alive_interval = 10)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################

# Tolerances for GMRES residual, see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
atol_lin_solve = 1e-3
rtol_lin_solve = 1e-3

# Jacobian-free Newton-Krylov (GMRES) solver
linsolve = KrylovJL_GMRES(atol = atol_lin_solve, rtol = rtol_lin_solve)

# Use Runge-Kutta method with Jacobian-free (!) Newton-Krylov (GMRES) implicit solver, see
# https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Using-Jacobian-Free-Newton-Krylov
ode_alg = IMEXEuler(autodiff = AutoFiniteDiff(), linsolve = linsolve)
sol = solve(ode, ode_alg; dt = 0.1, # Fixed timestep
            ode_default_options()..., callback = callbacks)
