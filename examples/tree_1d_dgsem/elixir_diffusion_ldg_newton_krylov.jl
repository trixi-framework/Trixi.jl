using Trixi
using OrdinaryDiffEqSDIRK
using LinearSolve # For Jacobian-free Newton-Krylov (GMRES) solver
using ADTypes # For automatic differentiation via finite differences

###############################################################################
# semidiscretization of the linear (advection) diffusion equation

advection_velocity = 0.0 # Note: This renders the equation mathematically purely parabolic
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.5
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

# surface flux does not matter for pure diffusion problem
solver = DGSEM(polydeg = 3, surface_flux = flux_central)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

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

solver_parabolic = ViscousFormulationLocalDG()
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver; solver_parabolic)

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

sol = solve(ode,
            # Use (diagonally) implicit Runge-Kutta method with Jacobian-free Newton-Krylov (GMRES) solver
            # See https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Using-Jacobian-Free-Newton-Krylov
            KenCarp47(autodiff = AutoFiniteDiff(), linsolve = KrylovJL_GMRES());
            ode_default_options()..., callback = callbacks)
