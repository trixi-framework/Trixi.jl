# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Abstract base type for time integration schemes of storage class `2N`
abstract type SimpleAlgorithm2N <: AbstractTimeIntegrationAlgorithm end

"""
    CarpenterKennedy2N54()

The following structures and methods provide a minimal implementation of
the low-storage explicit Runge-Kutta method of

- Carpenter, Kennedy (1994)
  Fourth-order 2N-storage Runge-Kutta schemes (Solution 3)
  URL: https://ntrs.nasa.gov/citations/19940028444
  File: https://ntrs.nasa.gov/api/citations/19940028444/downloads/19940028444.pdf

using the same interface as OrdinaryDiffEq.jl.
"""
struct CarpenterKennedy2N54 <: SimpleAlgorithm2N
    a::SVector{5, Float64}
    b::SVector{5, Float64}
    c::SVector{5, Float64}

    function CarpenterKennedy2N54()
        a = SVector(0.0,
                    567301805773.0 / 1357537059087.0,
                    2404267990393.0 / 2016746695238.0,
                    3550918686646.0 / 2091501179385.0,
                    1275806237668.0 / 842570457699.0)
        b = SVector(1432997174477.0 / 9575080441755.0,
                    5161836677717.0 / 13612068292357.0,
                    1720146321549.0 / 2090206949498.0,
                    3134564353537.0 / 4481467310338.0,
                    2277821191437.0 / 14882151754819.0)
        c = SVector(0.0,
                    1432997174477.0 / 9575080441755.0,
                    2526269341429.0 / 6820363962896.0,
                    2006345519317.0 / 3224310063776.0,
                    2802321613138.0 / 2924317926251.0)

        new(a, b, c)
    end
end

"""
    CarpenterKennedy2N43()

The following structures and methods provide a minimal implementation of
the low-storage explicit Runge-Kutta method of

- Carpenter, Kennedy (1994)
  Third-order 2N-storage Runge-Kutta schemes with error control
  URL: https://ntrs.nasa.gov/citations/19940028444
  File: https://ntrs.nasa.gov/api/citations/19940028444/downloads/19940028444.pdf

using the same interface as OrdinaryDiffEq.jl.
"""
struct CarpenterKennedy2N43 <: SimpleAlgorithm2N
    a::SVector{4, Float64}
    b::SVector{4, Float64}
    c::SVector{4, Float64}

    function CarpenterKennedy2N43()
        a = SVector(0, 756391 / 934407, 36441873 / 15625000, 1953125 / 1085297)
        b = SVector(8 / 141, 6627 / 2000, 609375 / 1085297, 198961 / 526383)
        c = SVector(0, 8 / 141, 86 / 125, 1)

        new(a, b, c)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleIntegratorOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    const adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    const maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SimpleIntegratorOptions(callback, tspan; maxiters = typemax(Int), kwargs...)
    SimpleIntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters,
                                              [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct SimpleIntegrator2N{RealT <: Real, uType <: AbstractVector,
                                  Params, Sol, F, Alg,
                                  SimpleIntegratorOptions} <: AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    const f::F # `rhs!` of the semidiscretization
    const alg::Alg # SimpleAlgorithm2N
    opts::SimpleIntegratorOptions
    finalstep::Bool # added for convenience
end

function init(ode::ODEProblem, alg::SimpleAlgorithm2N;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    u_tmp = similar(u)
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleIntegrator2N(u, du, u_tmp, t, dt, zero(dt), iter, ode.p,
                                    (prob = ode,), ode.f, alg,
                                    SimpleIntegratorOptions(callback, ode.tspan;
                                                            kwargs...), false)

    initialize_callbacks!(callback, integrator)

    return integrator
end

function step!(integrator::SimpleIntegrator2N)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    limit_dt!(integrator, t_end)

    # one time step
    integrator.u_tmp .= 0
    for stage in eachindex(alg.c)
        t_stage = integrator.t + integrator.dt * alg.c[stage]
        integrator.f(integrator.du, integrator.u, prob.p, t_stage)

        a_stage = alg.a[stage]
        b_stage_dt = alg.b[stage] * integrator.dt
        @trixi_timeit timer() "Runge-Kutta step" begin
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.du[i] -
                                      integrator.u_tmp[i] * a_stage
                integrator.u[i] += integrator.u_tmp[i] * b_stage_dt
            end
        end
    end
    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)

    return nothing
end

# get a cache where the RHS can be stored
get_tmp_cache(integrator::SimpleIntegrator2N) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleIntegrator2N, ::Bool) = false

# stop the time integration
function terminate!(integrator::SimpleIntegrator2N)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)

    return nothing
end

# used for AMR
function Base.resize!(integrator::SimpleIntegrator2N, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    return nothing
end
end # @muladd
