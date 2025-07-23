# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function integrate_w_dot_stage(stage, u_stage,
                                       mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                       equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_stage, equations, dg,
                                                i, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function integrate_w_dot_stage(stage, u_stage,
                                       mesh::Union{TreeMesh{3}, StructuredMesh{3},
                                                   P4estMesh{3}, T8codeMesh{3}},
                                       equations, dg::DG, cache)
    @trixi_timeit timer() "Integrate w ⋅ k" begin
        # Calculate ∫(∂S/∂u ⋅ k)dΩ = ∫(w ⋅ k)dΩ
        integrate_via_indices(u_stage, mesh, equations, dg, cache,
                              stage) do u_stage, i, j, k, element, equations, dg, stage
            w_node = cons2entropy(get_node_vars(u_stage, equations, dg,
                                                i, j, k, element),
                                  equations)
            stage_node = get_node_vars(stage, equations, dg, i, j, k, element)
            dot(w_node, stage_node)
        end
    end
end

@inline function entropy_difference(gamma, S_old, dS, u_gamma_dir, mesh,
                                    equations, dg::DG, cache)
    return integrate(entropy, u_gamma_dir, mesh, equations, dg, cache) -
           S_old - gamma * dS # `dS` is true entropy change computed from stages
end

@inline function add_direction!(u_tmp_wrap, u_wrap, dir_wrap, gamma,
                                dg::DG, cache)
    @threaded for element in eachelement(dg, cache)
        @views @. u_tmp_wrap[.., element] = u_wrap[.., element] +
                                            gamma * dir_wrap[.., element]
    end
end

"""
    AbstractRelaxationSolver

Abstract type for relaxation solvers used to compute the relaxation parameter `` \\gamma`` 
in the entropy relaxation time integration method [`SubDiagonalRelaxationAlgorithm`](@ref).
Implemented solver is currently [`RelaxationSolverNewton`](@ref).
"""
abstract type AbstractRelaxationSolver end

@doc raw"""
    RelaxationSolverNewton(; max_iterations = 5,
                             root_tol = 1e-15, gamma_tol = 1e-13,
                             gamma_min = 1e-13, step_scaling = 1.0)

Solve the relaxation equation 
```math
\eta \big(\boldsymbol U_{n+1}(\gamma_n) \big) = 
\eta \left( \boldsymbol U_n + \Delta t \gamma_n \sum_{i=1}^Sb_i \boldsymbol K_i  \right) \overset{!}{=} 
\eta(\boldsymbol U_n) + \gamma_n \Delta \eta (\boldsymbol U_n)
```
with true entropy change
```math
\Delta \eta \coloneqq 
\Delta t \sum_{i=1}^S b_i 
\left \langle \frac{\partial \eta(\boldsymbol U_{n,i})}{\partial \boldsymbol U_{n,i}}, 
\boldsymbol K_i 
\right \rangle	
```
for the relaxation parameter ``\gamma_n`` using Newton's method.
The derivative of the relaxation function is known and can be directly computed.
Supposed to be supplied to a relaxation Runge-Kutta method such as [`SubDiagonalAlgorithm`](@ref).

# Arguments
- `max_iterations::Int`: Maximum number of Newton iterations.
- `root_tol::RealT`: Function-tolerance for the relaxation equation, i.e., 
                     the absolute defect of the left and right-hand side of the equation above, i.e.,
                     the solver stops if
                     ``|\eta_{n+1} - (\eta_n + \gamma_n \Delta \eta( \boldsymbol U_n))| \leq \text{root\_tol}``.
- `gamma_tol::RealT`: Absolute tolerance for the Newton update step size, i.e., the solver stops if 
                      ``|\gamma_{\text{new}} - \gamma_{\text{old}}| \leq \text{gamma\_tol}``.
- `gamma_min::RealT`: Minimum relaxation parameter. If the Newton iteration results a value smaller than this, 
                      the relaxation parameter is set to 1.
- `step_scaling::RealT`: Scaling factor for the Newton step. For `step_scaling > 1` the Newton procedure is accelerated, while for `step_scaling < 1` it is damped.
"""
struct RelaxationSolverNewton{RealT <: Real} <: AbstractRelaxationSolver
    # General parameters
    max_iterations::Int # Maximum number of Newton iterations
    root_tol::RealT     # Function-tolerance for the relaxation equation
    gamma_tol::RealT    # Absolute tolerance for the Newton update step size
    # Method-specific parameters
    # Minimum relaxation parameter. If the Newton iteration computes a value smaller than this, 
    # the relaxation parameter is set to 1.
    gamma_min::RealT
    step_scaling::RealT # Scaling factor for the Newton step
end
function RelaxationSolverNewton(; max_iterations = 5,
                                root_tol = 1e-15, gamma_tol = 1e-13,
                                gamma_min = 1e-13, step_scaling = 1.0)
    return RelaxationSolverNewton(max_iterations, root_tol, gamma_tol,
                                  gamma_min, step_scaling)
end

function Base.show(io::IO,
                   relaxation_solver::RelaxationSolverNewton)
    print(io, "RelaxationSolverNewton(max_iterations=",
          relaxation_solver.max_iterations,
          ", root_tol=", relaxation_solver.root_tol,
          ", gamma_tol=", relaxation_solver.gamma_tol,
          ", gamma_min=", relaxation_solver.gamma_min,
          ", step_scaling=", relaxation_solver.step_scaling, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   relaxation_solver::RelaxationSolverNewton)
    if get(io, :compact, false)
        show(io, relaxation_solver)
    else
        setup = [
            "max_iterations" => relaxation_solver.max_iterations,
            "root_tol" => relaxation_solver.root_tol,
            "gamma_tol" => relaxation_solver.gamma_tol,
            "gamma_min" => relaxation_solver.gamma_min,
            "step_scaling" => relaxation_solver.step_scaling
        ]
        summary_box(io, "RelaxationSolverNewton", setup)
    end
end

function relaxation_solver!(integrator,
                            u_tmp_wrap, u_wrap, dir_wrap, dS,
                            mesh, equations, dg::DG, cache,
                            relaxation_solver::RelaxationSolverNewton)
    @unpack max_iterations, root_tol, gamma_tol, gamma_min, step_scaling = relaxation_solver

    iterations = 0
    entropy_residual = 0
    while iterations < max_iterations
        add_direction!(u_tmp_wrap, u_wrap, dir_wrap, integrator.gamma, dg, cache)
        @trixi_timeit timer() "Δη" entropy_residual=entropy_difference(integrator.gamma,
                                                                       integrator.S_old,
                                                                       dS, u_tmp_wrap,
                                                                       mesh, equations,
                                                                       dg, cache)

        if abs(entropy_residual) <= root_tol # Sufficiently close at root
            break
        end

        # Derivative of object relaxation function `r` with respect to `gamma`
        dr = integrate_w_dot_stage(dir_wrap, u_tmp_wrap, mesh, equations, dg, cache) -
             dS

        step = step_scaling * entropy_residual / dr # Newton-Raphson update step
        if abs(step) <= gamma_tol # Prevent unnecessary small steps
            break
        end

        integrator.gamma -= step # Perform Newton-Raphson update
        iterations += 1
    end

    # Catch Newton failures
    if integrator.gamma < gamma_min || isnan(integrator.gamma) ||
       isinf(integrator.gamma)
        integrator.gamma = 1
        entropy_residual = 0 # May be very large, avoid using this in `S_old`
    end
    # Update old entropy
    integrator.S_old += integrator.gamma * dS + entropy_residual

    return nothing
end
end # @muladd
