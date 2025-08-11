# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    vanderHouwenAlgorithm

Abstract type for sub-diagonal Runge-Kutta methods, i.e., 
methods with a Butcher tableau of the form
```math
\begin{array}
    {c|c|c c c c c c}
    i & \boldsymbol c & & & A & & & \\
    \hline
    1 & 0 & & & & & & \\
    2 & c_2 & a_{21} & & & & & \\
    3 & c_3 & b_1 & a_{32} & & & & \\ 
    4 & c_4 & b_1 & b_2 & a_{43} & & & \\ 
    \vdots & \vdots & \vdots & \vdots & \ddots & \ddots & & \\
    S & c_S & b_1 & b_2 & \dots & b_{S-2} & a_{S, S-1} & \\
    \hline
    & & b_1 & b_2 & \dots & b_{S-2} & b_{S-1} & b_S
\end{array}
```

Currently implemented methods are the Carpenter-Kennedy-Lewis 4-stage, 3rd-order method [`CKL43`](@ref)
and the Carpenter-Kennedy-Lewis 5-stage, 4th-order method [`CKL54`](@ref) which are optimized for the 
compressible Navier-Stokes equations.
"""
abstract type vanderHouwenAlgorithm <: AbstractTimeIntegrationAlgorithm end

"""
    vanderHouwenRelaxationAlgorithm

Abstract type for van-der-Houwen type Runge-Kutta algorithms (see [`vanderHouwenAlgorithm`](@ref)) 
with relaxation to achieve entropy-conservation/stability.
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

Currently implemented methods are the Carpenter-Kennedy-Lewis 4-stage, 3rd-order method [`RelaxationCKL43`](@ref)
and the Carpenter-Kennedy-Lewis 5-stage, 4th-order method [`RelaxationCKL54`](@ref) which are optimized for the 
compressible Navier-Stokes equations.
"""
abstract type vanderHouwenRelaxationAlgorithm <:
              AbstractRelaxationTimeIntegrationAlgorithm end

"""
    CKL43()

Carpenter-Kennedy-Lewis 4-stage, 3rd-order low-storage Runge-Kutta method,
optimized for the compressible Navier-Stokes equations.
Implemented as a [`vanderHouwenAlgorithm`](@ref).
For the exact coefficients consult the original paper:

- Kennedy, Carpenter, Lewis (2000)
  Low-storage, explicit Runge-Kutta schemes for the compressible Navier-Stokes equations
  [DOI: 10.1016/S0168-9274(99)00141-5](https://doi.org/10.1016/S0168-9274(99)00141-5)
"""
struct CKL43 <: vanderHouwenAlgorithm
    a::SVector{4, Float64}
    b::SVector{4, Float64}
    c::SVector{4, Float64}
end
function CKL43()
    a = SVector(0.0,
                11847461282814 / 36547543011857,
                3943225443063 / 7078155732230,
                -346793006927 / 4029903576067)

    b = SVector(1017324711453 / 9774461848756,
                8237718856693 / 13685301971492,
                57731312506979 / 19404895981398,
                -101169746363290 / 37734290219643)
    c = SVector(0.0,
                a[2],
                b[1] + a[3],
                b[1] + b[2] + a[4])

    return CKL43(a, b, c)
end

"""
    RelaxationCKL43(; relaxation_solver = RelaxationSolverNewton())

Relaxation version of the 4-stage, 3rd-order low-storage Runge-Kutta method [`CKL43()`](@ref), 
implemented as a [`vanderHouwenRelaxationAlgorithm`](@ref).
The default relaxation solver [`AbstractRelaxationSolver`](@ref) is [`RelaxationSolverNewton`](@ref).
"""
struct RelaxationCKL43{AbstractRelaxationSolver} <: vanderHouwenRelaxationAlgorithm
    van_der_houwen_alg::CKL43
    relaxation_solver::AbstractRelaxationSolver
end
function RelaxationCKL43(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationCKL43{typeof(relaxation_solver)}(CKL43(), relaxation_solver)
end

"""
    CKL54()

Carpenter-Kennedy-Lewis 5-stage, 4th-order low-storage Runge-Kutta method,
optimized for the compressible Navier-Stokes equations.
Implemented as a [`vanderHouwenAlgorithm`](@ref).
For the exact coefficients consult the original paper:

- Kennedy, Carpenter, Lewis (2000)
  Low-storage, explicit Runge-Kutta schemes for the compressible Navier-Stokes equations
  [DOI: 10.1016/S0168-9274(99)00141-5](https://doi.org/10.1016/S0168-9274(99)00141-5)
"""
struct CKL54 <: vanderHouwenAlgorithm
    a::SVector{5, Float64}
    b::SVector{5, Float64}
    c::SVector{5, Float64}
end
function CKL54()
    a = SVector(0.0,
                970286171893 / 4311952581923,
                6584761158862 / 12103376702013,
                2251764453980 / 15575788980749,
                26877169314380 / 34165994151039)

    b = SVector(1153189308089 / 22510343858157,
                1772645290293 / 4653164025191,
                -1672844663538 / 4480602732383,
                2114624349019 / 3568978502595,
                5198255086312 / 14908931495163)
    c = SVector(0.0,
                a[2],
                b[1] + a[3],
                b[1] + b[2] + a[4],
                b[1] + b[2] + b[3] + a[5])

    return CKL54(a, b, c)
end

"""
    RelaxationCKL54(; relaxation_solver = RelaxationSolverNewton())

Relaxation version of the 4-stage, 3rd-order low-storage Runge-Kutta method [`CKL54()`](@ref), 
implemented as a [`vanderHouwenRelaxationAlgorithm`](@ref).
The default relaxation solver [`AbstractRelaxationSolver`](@ref) is [`RelaxationSolverNewton`](@ref).
"""
struct RelaxationCKL54{AbstractRelaxationSolver} <: vanderHouwenRelaxationAlgorithm
    van_der_houwen_alg::CKL54
    relaxation_solver::AbstractRelaxationSolver
end
function RelaxationCKL54(; relaxation_solver = RelaxationSolverNewton())
    return RelaxationCKL54{typeof(relaxation_solver)}(CKL54(), relaxation_solver)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct vanderHouwenRelaxationIntegrator{RealT <: Real, uType, Params, Sol, F,
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
    alg::Alg # `vanderHouwenRelaxationAlgorithm`
    opts::SimpleIntegratorOptions
    finalstep::Bool # added for convenience
    # Addition for efficient implementation
    k_prev::uType
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

function init(ode::ODEProblem, alg::vanderHouwenRelaxationAlgorithm;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    u_tmp = copy(u)
    k_prev = similar(u)

    t = first(ode.tspan)
    iter = 0

    # For entropy relaxation
    direction = zero(u)
    gamma = one(eltype(u))
    semi = ode.p
    u_wrap = wrap_array(u, semi)
    S_old = integrate(entropy, u_wrap, semi.mesh, semi.equations, semi.solver,
                      semi.cache)

    integrator = vanderHouwenRelaxationIntegrator(u, du, u_tmp, t, dt, zero(dt), iter,
                                                  ode.p, (prob = ode,), ode.f,
                                                  alg.van_der_houwen_alg,
                                                  SimpleIntegratorOptions(callback,
                                                                          ode.tspan;
                                                                          kwargs...),
                                                  false,
                                                  k_prev, direction, gamma, S_old,
                                                  alg.relaxation_solver)

    initialize_callbacks!(callback, integrator)

    return integrator
end

function step!(integrator::vanderHouwenRelaxationIntegrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    limit_dt!(integrator)

    @trixi_timeit timer() "Relaxation vdH RK integration step" begin
        num_stages = length(alg.c)

        mesh, equations, dg, cache = mesh_equations_solver_cache(prob.p)
        u_wrap = wrap_array(integrator.u, prob.p)
        u_tmp_wrap = wrap_array(integrator.u_tmp, prob.p)

        # First stage
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        # Try to enable optimizations due to `muladd` by computing this factor only once, see
        # https://github.com/trixi-framework/Trixi.jl/pull/2480#discussion_r2224529532
        b1dt = alg.b[1] * integrator.dt
        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = b1dt * integrator.du[i]

            integrator.k_prev[i] = integrator.du[i] # Faster than broadcasted version (with .=)
        end

        du_wrap = wrap_array(integrator.du, prob.p)
        # Entropy change due to first stage
        dS = alg.b[1] * integrator.dt *
             integrate_w_dot_stage(du_wrap, u_wrap, mesh, equations, dg, cache)

        a2_dt = alg.a[2] * integrator.dt
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] + a2_dt * integrator.du[i]
        end

        # Second to last stage
        for stage in 2:(num_stages - 1)
            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            # Entropy change due to current stage
            bs_dt = alg.b[stage] * integrator.dt
            dS += bs_dt *
                  integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

            bsminus1_minus_as = alg.b[stage - 1] - alg.a[stage]
            @threaded for i in eachindex(integrator.u)
                # Try to enable optimizations due to `muladd` by avoiding `+=`
                # https://github.com/trixi-framework/Trixi.jl/pull/2480#discussion_r2224531702
                integrator.direction[i] = integrator.direction[i] +
                                          bs_dt * integrator.du[i]

                # Subtract previous stage contribution from `u_tmp` and add most recent one
                integrator.u_tmp[i] = integrator.u_tmp[i] +
                                      integrator.dt *
                                      (bsminus1_minus_as * integrator.k_prev[i] +
                                       alg.a[stage + 1] * integrator.du[i])

                integrator.k_prev[i] = integrator.du[i] # Faster than broadcasted version (with .=)
            end
        end

        # Last stage
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[num_stages] * integrator.dt)

        bs_dt = alg.b[num_stages] * integrator.dt
        dS += bs_dt *
              integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

        @threaded for i in eachindex(integrator.u)
            integrator.direction[i] = integrator.direction[i] + bs_dt * integrator.du[i]
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
            integrator.u[i] = integrator.u[i] +
                              integrator.gamma * integrator.direction[i]
        end
    end

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)

    return nothing
end

# used for AMR
function Base.resize!(integrator::vanderHouwenRelaxationIntegrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)
    resize!(integrator.k_prev, new_size)
    # Relaxation addition
    resize!(integrator.direction, new_size)

    return nothing
end
end # @muladd
