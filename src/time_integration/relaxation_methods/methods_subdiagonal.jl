# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    SubDiagonalAlgorithm

Abstract type for sub-diagonal Runge-Kutta methods, i.e., 
methods with a Butcher tableau of the form
```math
\begin{array}
    {c|c|c c c c c}
    i & \boldsymbol c & & & A & & \\
    \hline
    1 & 0 & & & & &\\
    2 & c_2 & c_2 & & & &  \\
    3 & c_3 & 0 & c_3 & & &  \\ 
    4 & c_4 & 0 & 0 & c_4 & \\
    \vdots & & \vdots & \vdots & \ddots & \ddots &  \\
    S & c_S & 0 & & \dots & 0 & c_S \\
    \hline
    & & b_1 & b_2 & \dots & & b_S
\end{array}
```

Currently implemented are the third-order, three-stage method by Ralston [`RK33`](@ref) 
and the canonical fourth-order, four-stage method by Kutta [`RK44`](@ref).
"""
abstract type SubDiagonalAlgorithm <: AbstractTimeIntegrationAlgorithm end

"""
    SubDiagonalRelaxationAlgorithm

Abstract type for sub-diagonal Runge-Kutta algorithms (see [`SubDiagonalAlgorithm`](@ref)) 
with relaxation to achieve entropy conservation/stability.
In addition to the standard Runge-Kutta method, these algorithms are equipped with a
relaxation solver [`AbstractRelaxationSolver`](@ref) which is used to compute the relaxation parameter ``\\gamma``.
This allows the relaxation methods to suppress entropy defects due to the time stepping.

For details on the relaxation procedure, see
- Ketcheson (2019)
  Relaxation Runge-Kutta Methods: Conservation and Stability for Inner-Product Norms
  [DOI: 10.1137/19M1263662](https://doi.org/10.1137/19M1263662)
- Ranocha et al. (2020)
  Relaxation Runge-Kutta Methods: Fully Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier-Stokes Equations  
  [DOI: 10.1137/19M1263480](https://doi.org/10.1137/19M1263480)

Currently implemented are the third-order, three-stage method by Ralston [`RK33`](@ref) 
and the canonical fourth-order, four-stage method by Kutta [`RK44`](@ref).
"""
abstract type SubDiagonalRelaxationAlgorithm <: AbstractTimeIntegrationAlgorithm end

"""
    RK33()

Relaxation version of Ralston's third-order Runge-Kutta method, implemented as a [`SubDiagonalAlgorithm`](@ref).
The weight vector is given by ``\\boldsymbol b = [2/9, 1/3, 4/9]`` and the 
abscissae/timesteps by ``\\boldsymbol c = [0.0, 0.5, 0.75]``.

This method has minimum local error bound among the ``S=p=3`` methods.
- Ralston (1962)
  Runge-Kutta Methods with Minimum Error Bounds
  [DOI: 10.1090/S0025-5718-1962-0150954-0](https://doi.org/10.1090/S0025-5718-1962-0150954-0)
"""
struct RK33 <: SubDiagonalAlgorithm
    b::SVector{3, Float64}
    c::SVector{3, Float64}
end
function RK33()
    b = SVector(2 / 9, 1 / 3, 4 / 9)
    c = SVector(0.0, 0.5, 0.75)

    return RK33(b, c)
end

"""
    RelaxationRK33(; relaxation_solver = RelaxationSolverNewton())

Relaxation version of Ralston's third-order Runge-Kutta method [`RK33()`](@ref), 
implemented as a [`SubDiagonalRelaxationAlgorithm`](@ref).
The default relaxation solver [`AbstractRelaxationSolver`](@ref) is [`RelaxationSolverNewton`](@ref).
"""
struct RelaxationRK33{AbstractRelaxationSolver} <: SubDiagonalRelaxationAlgorithm
    sub_diagonal_alg::RK33
    relaxation_solver::AbstractRelaxationSolver
end
function RelaxationRK33(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationRK33{typeof(relaxation_solver)}(RK33(), relaxation_solver)
end

"""
    RK44()

The canonical fourth-order Runge-Kutta method, implemented as a [`SubDiagonalAlgorithm`](@ref).
The weight vector is given by ``\\boldsymbol b = [1/6, 1/3, 1/3, 1/6]`` and the 
abscissae/timesteps by ``\\boldsymbol c = [0.0, 0.5, 0.5, 1.0]``.
"""
struct RK44 <: SubDiagonalAlgorithm
    b::SVector{4, Float64}
    c::SVector{4, Float64}
end
function RK44()
    b = SVector(1 / 6, 1 / 3, 1 / 3, 1 / 6)
    c = SVector(0.0, 0.5, 0.5, 1.0)

    return RK44(b, c)
end

"""
    RelaxationRK44(; relaxation_solver = RelaxationSolverNewton())

Relaxation version of the canonical fourth-order Runge-Kutta method [`RK44()`](@ref), 
implemented as a [`SubDiagonalRelaxationAlgorithm`](@ref).
The default relaxation solver [`AbstractRelaxationSolver`](@ref) is [`RelaxationSolverNewton`](@ref).
"""
struct RelaxationRK44{AbstractRelaxationSolver} <: SubDiagonalRelaxationAlgorithm
    sub_diagonal_alg::RK44
    relaxation_solver::AbstractRelaxationSolver
end
function RelaxationRK44(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationRK44{typeof(relaxation_solver)}(RK44(), relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct SubDiagonalRelaxationIntegrator{RealT <: Real, uType, Params, Sol, F,
                                               Alg, SimpleIntegratorOptions,
                                               AbstractRelaxationSolver} <:
               RelaxationIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs` of the semidiscretization
    alg::Alg # `SubDiagonalRelaxationAlgorithm`
    opts::SimpleIntegratorOptions
    finalstep::Bool # added for convenience
    # Addition for Relaxation methodology
    direction::uType # RK update, i.e., sum of stages K_i times weights b_i
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::AbstractRelaxationSolver
    # Note: Could add another register which would store the summed-up 
    # dot products ∑ₖ (wₖ ⋅ kₖ) and then integrate only once and not per stage k
    # Could also add option `recompute_entropy` for entropy-conservative problems
    # to save redundant computations.
end

function init(ode::ODEProblem, alg::SubDiagonalRelaxationAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = zero(u)
    direction = zero(u)

    t = first(ode.tspan)
    iter = 0

    # For entropy relaxation
    gamma = one(eltype(u))

    semi = ode.p
    u_wrap = wrap_array(u, semi)
    S_old = integrate(entropy, u_wrap, semi.mesh, semi.equations, semi.solver,
                      semi.cache)

    integrator = SubDiagonalRelaxationIntegrator(u, du, u_tmp, t, dt, zero(dt), iter,
                                                 ode.p, (prob = ode,), ode.f,
                                                 alg.sub_diagonal_alg,
                                                 SimpleIntegratorOptions(callback,
                                                                         ode.tspan;
                                                                         kwargs...),
                                                 false,
                                                 direction, gamma, S_old,
                                                 alg.relaxation_solver)

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with sub-diagonal time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem,
               alg::SubDiagonalRelaxationAlgorithm;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function step!(integrator::SubDiagonalRelaxationIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Relaxation sub-diagonal RK integration step" begin
        mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)

        u_wrap = wrap_array(integrator.u, prob.p)
        u_tmp_wrap = wrap_array(integrator.u_tmp, prob.p)

        # First stage
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        b1_dt = alg.b[1] * integrator.dt
        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = b1_dt * integrator.du[i]
        end

        du_wrap = wrap_array(integrator.du, prob.p)
        # Entropy change due to first stage
        dS = alg.b[1] * integrator.dt *
             integrate_w_dot_stage(du_wrap, u_wrap, mesh, equations, dg, cache)

        # Second to last stage
        for stage in 2:length(alg.c)
            c_dt = alg.c[stage] * integrator.dt
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] + c_dt * integrator.du[i]
            end
            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)
            b_dt = alg.b[stage] * integrator.dt
            @threaded for i in eachindex(integrator.u)
                integrator.direction[i] = integrator.direction[i] + b_dt * integrator.du[i]
            end

            # Entropy change due to current stage
            dS += alg.b[stage] * integrator.dt *
                  integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)
        end

        direction_wrap = wrap_array(integrator.direction, prob.p)

        @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                     u_tmp_wrap, u_wrap,
                                                                     direction_wrap, dS,
                                                                     mesh, equations,
                                                                     dg, cache,
                                                                     integrator.relaxation_solver)

        integrator.iter += 1
        update_t_relaxation!(integrator)

        # Do relaxed update
        @threaded for i in eachindex(integrator.u)
            integrator.u[i] = integrator.u[i] + integrator.gamma * integrator.direction[i]
        end
    end

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

# used for AMR
function Base.resize!(integrator::SubDiagonalRelaxationIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    # Relaxation addition
    resize!(integrator.direction, new_size)
end
end # @muladd
