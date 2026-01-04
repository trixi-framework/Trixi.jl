using OrdinaryDiffEqSDIRK
using SciMLBase, SciMLOperators, SparseArrays
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 0.5f0
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true,
                n_cells_max = 30_000)

function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation2D)
    # Store translated coordinate for easy use of exact solution
    RealT = eltype(x)
    x_trans = x - equation.advection_velocity * t

    nu = diffusivity()
    c = 1
    A = 0.5f0
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE setup

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan) # For accessing the initial condition u0 only

D_map, _ = linear_structure_parabolic(semi)
# Cannot directly construct `MatrixOperator` from `LinearMap`, need detour via sparse matrix
D_op = MatrixOperator(sparse(D_map))

split_func = SplitFunction(D_op, Trixi.rhs!)
ode_operator = SplitODEProblem{true}(split_func, ode.u0, tspan, semi)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode_operator, KenCarp4();
            adaptive = false, dt = 1e-2,
            ode_default_options()..., callback = callbacks)
