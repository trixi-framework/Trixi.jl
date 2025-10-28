using Trixi

###############################################################################

# Build pure diffusion (Laplace) operator
advection_velocity = (0, 0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 1
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# The hyperbolic flux does not matter for this example since
# the hyperbolic part is zero.
solver = DGSEM(polydeg = 5, surface_flux = flux_central)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 80_000,
                periodicity = false)

# Analytical/continuous steady-state solution
function continuous_solution(x, t, equations)
    a = (1 - cosh(2 * pi)) / (sinh(2 * pi))

    u_sol = (cosh(2 * pi * x[2]) + a * sinh(2 * pi * x[2])) * sinpi(2 * x[1])
    return SVector(u_sol)
end
initial_condition = continuous_solution

function bc_homogeneous(x, t, equations)
    return SVector(0)
end
bc_homogeneous_dirichlet = BoundaryConditionDirichlet(bc_homogeneous)

function bc_sin(x, t, equations)
    return SVector(sinpi(2 * x[1]))
end
bc_sin_dirichlet = BoundaryConditionDirichlet(bc_sin)

# Same boundary conditions for hyperbolic and parabolic part
boundary_conditions = (; x_neg = bc_homogeneous_dirichlet,
                       y_neg = bc_sin_dirichlet,
                       y_pos = bc_sin_dirichlet,
                       x_pos = bc_homogeneous_dirichlet)

# `solver_parabolic = ViscousFormulationLocalDG()` strictly required for elliptic/diffusion-dominated problem
semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationLocalDG(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions))

# Note that `linear_structure` does not access the `initial_condition`/steady-state solution
A_map, b = linear_structure(semi)

# Direct solve, with explicit sparse matrix construction.
# Has trouble due to poor conditioning, visible in top right corner
#=
using SparseArrays
A_matrix = sparse(A_map)
u_ls = A_matrix \ b
=#

# Iterative solve, works directly on the linear map, no explicit matrix construction required!
using Krylov

atol = 1e-11
rtol = 1e-10

# This solves the Laplace equation (i.e., steady-state diffusion/heat equation).
# Note that we do not use CG since the operator `A_map` itself is only
# symmetric and positive definite with respect to the inner product induced by
# the DGSEM quadrature, not the standard Euclidean inner product. Standard CG
# would require multiplying the result of `A_map * u` (and `b`) by the mass matrix.
u_ls, stats = gmres(A_map, b, atol = atol, rtol = rtol)

###############################################################################

# Construct the ODE problem for easy plotting and comparison to analytical solution
# Note that we only solve a steady-state problem, i.e., `tspan` has a range of zero.
tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
# Analysis callback quantifies discretization/interpolation error of the exact solution
analysis_callback = AnalysisCallback(semi)
callbacks = CallbackSet(summary_callback,
                        analysis_callback)

# Choice of ODE Solver does not matter here
using OrdinaryDiffEqLowStorageRK

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1e-4,
            ode_default_options()..., callback = callbacks);

# Check interpolation errors due to choice of number of cells & polynomial degree
interpolation_errors = analysis_callback(sol)

using Plots
# Plot analytical solution (interpolated initial condition)
plot(sol)

# Inject linear system solution for plotting & error computation
sol.u[end] = u_ls
plot(sol)

# Check linear system solution errors
linear_solution_errors = analysis_callback(sol)
