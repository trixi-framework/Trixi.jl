# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Abstract base type for time integration schemes of explicit strong stability-preserving (SSP)
# Runge-Kutta (RK) methods. They are high-order time discretizations that guarantee the TVD property.
abstract type SimpleAlgorithmSSP end

"""
    SimpleSSPRK33(; stage_callbacks=())

The third-order SSP Runge-Kutta method of Shu and Osher.

## References

- Shu, Osher (1988)
  "Efficient Implementation of Essentially Non-oscillatory Shock-Capturing Schemes" (Eq. 2.18)
  [DOI: 10.1016/0021-9991(88)90177-5](https://doi.org/10.1016/0021-9991(88)90177-5)
"""
struct SimpleSSPRK33{StageCallbacks} <: SimpleAlgorithmSSP
    numerator_a::SVector{3, Float64}
    numerator_b::SVector{3, Float64}
    denominator::SVector{3, Float64}
    c::SVector{3, Float64}
    stage_callbacks::StageCallbacks

    function SimpleSSPRK33(; stage_callbacks = ())
        # Mathematically speaking, it is not necessary for the algorithm to split the factors
        # into numerator and denominator. Otherwise, however, rounding errors of the order of
        # the machine accuracy will occur, which will add up over time and thus endanger the
        # conservation of the simulation.
        # See also https://github.com/trixi-framework/Trixi.jl/pull/1640.
        numerator_a = SVector(0.0, 3.0, 1.0) # a = numerator_a / denominator
        numerator_b = SVector(1.0, 1.0, 2.0) # b = numerator_b / denominator
        denominator = SVector(1.0, 4.0, 3.0)
        c = SVector(0.0, 1.0, 1 / 2)

        # Butcher tableau
        #   c |       a
        #   0 |
        #   1 |   1
        # 1/2 | 1/4  1/4
        # --------------------
        #   b | 1/6  1/6  2/3

        new{typeof(stage_callbacks)}(numerator_a, numerator_b, denominator, c,
                                     stage_callbacks)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleIntegratorSSPOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SimpleIntegratorSSPOptions(callback, tspan; maxiters = typemax(Int), kwargs...)
    tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
    # We add last(tspan) to make sure that the time integration stops at the end time
    push!(tstops_internal, last(tspan))
    # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
    # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
    push!(tstops_internal, 2 * last(tspan))
    SimpleIntegratorSSPOptions{typeof(callback), typeof(tstops_internal)}(callback,
                                                                          false, Inf,
                                                                          maxiters,
                                                                          tstops_internal)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct SimpleIntegratorSSP{RealT <: Real, uType, Params, Sol, F, Alg,
                                   SimpleIntegratorSSPOptions}
    u::uType
    du::uType
    r0::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg
    opts::SimpleIntegratorSSPOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
end

"""
    add_tstop!(integrator::SimpleIntegratorSSP, t)
Add a time stop during the time integration process.
This function is called after the periodic SaveSolutionCallback to specify the next stop to save the solution.
"""
function add_tstop!(integrator::SimpleIntegratorSSP, t)
    integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
        error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
    # We need to remove the first entry of tstops when a new entry is added.
    # Otherwise, the simulation gets stuck at the previous tstop and dt is adjusted to zero.
    if length(integrator.opts.tstops) > 1
        pop!(integrator.opts.tstops)
    end
    push!(integrator.opts.tstops, integrator.tdir * t)
end

has_tstop(integrator::SimpleIntegratorSSP) = !isempty(integrator.opts.tstops)
first_tstop(integrator::SimpleIntegratorSSP) = first(integrator.opts.tstops)

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleIntegratorSSP, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

"""
    solve(ode, alg; dt, callbacks, kwargs...)

The following structures and methods provide the infrastructure for SSP Runge-Kutta methods
of type `SimpleAlgorithmSSP`.
"""
function solve(ode::ODEProblem, alg = SimpleSSPRK33()::SimpleAlgorithmSSP;
               dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    r0 = similar(u)
    t = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0
    integrator = SimpleIntegratorSSP(u, du, r0, t, tdir, dt, dt, iter, ode.p,
                                     (prob = ode,), ode.f, alg,
                                     SimpleIntegratorSSPOptions(callback, ode.tspan;
                                                                kwargs...),
                                     false, true, false)

    # resize container
    resize!(integrator.p, nelements(integrator.p.solver, integrator.p.cache))

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with the SSP time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    for stage_callback in alg.stage_callbacks
        init_callback(stage_callback, integrator.p)
    end

    solve!(integrator)
end

function solve!(integrator::SimpleIntegratorSSP)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    integrator.finalstep = false
    @trixi_timeit timer() "main loop" while !integrator.finalstep
        if isnan(integrator.dt)
            error("time step size `dt` is NaN")
        end

        modify_dt_for_tstops!(integrator)

        # if the next iteration would push the simulation beyond the end time, set dt accordingly
        if integrator.t + integrator.dt > t_end ||
           isapprox(integrator.t + integrator.dt, t_end)
            integrator.dt = t_end - integrator.t
            terminate!(integrator)
        end

        @. integrator.r0 = integrator.u
        for stage in eachindex(alg.c)
            t_stage = integrator.t + integrator.dt * alg.c[stage]
            # compute du
            integrator.f(integrator.du, integrator.u, integrator.p, t_stage)

            # perform forward Euler step
            @. integrator.u = integrator.u + integrator.dt * integrator.du

            for stage_callback in alg.stage_callbacks
                stage_callback(integrator.u, integrator, stage)
            end

            # perform convex combination
            @. integrator.u = (alg.numerator_a[stage] * integrator.r0 +
                               alg.numerator_b[stage] * integrator.u) /
                              alg.denominator[stage]
        end

        integrator.iter += 1
        integrator.t += integrator.dt

        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
            end
        end

        # respect maximum number of iterations
        if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
            @warn "Interrupted. Larger maxiters is needed."
            terminate!(integrator)
        end
    end

    # Empty the tstops array.
    # This cannot be done in terminate!(integrator::SimpleIntegratorSSP) because DiffEqCallbacks.PeriodicCallbackAffect would return at error.
    extract_all!(integrator.opts.tstops)

    for stage_callback in alg.stage_callbacks
        finalize_callback(stage_callback, integrator.p)
    end

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u), prob)
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegratorSSP) = integrator.du
get_tmp_cache(integrator::SimpleIntegratorSSP) = (integrator.r0,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleIntegratorSSP, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleIntegratorSSP, dt)
    (integrator.dt = dt; integrator.dtcache = dt)
end

# used by adaptive timestepping algorithms in DiffEq
function get_proposed_dt(integrator::SimpleIntegratorSSP)
    return ifelse(integrator.opts.adaptive, integrator.dt, integrator.dtcache)
end

# stop the time integration
function terminate!(integrator::SimpleIntegratorSSP)
    integrator.finalstep = true
end

"""
    modify_dt_for_tstops!(integrator::SimpleIntegratorSSP)
Modify the time-step size to match the time stops specified in integrator.opts.tstops.
To avoid adding OrdinaryDiffEq to Trixi's dependencies, this routine is a copy of
https://github.com/SciML/OrdinaryDiffEq.jl/blob/d76335281c540ee5a6d1bd8bb634713e004f62ee/src/integrators/integrator_utils.jl#L38-L54
"""
function modify_dt_for_tstops!(integrator::SimpleIntegratorSSP)
    if has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = first_tstop(integrator)
        if integrator.opts.adaptive
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif iszero(integrator.dtcache) && integrator.dtchangeable
            integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
        elseif integrator.dtchangeable && !integrator.force_stepfail
            # always try to step! with dtcache, but lower if a tstop
            # however, if force_stepfail then don't set to dtcache, and no tstop worry
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dtcache), abs(tdir_tstop - tdir_t)) # step! to the end
        end
    end
end

# used for AMR
function Base.resize!(integrator::SimpleIntegratorSSP, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.r0, new_size)

    # Resize container
    # new_size = n_variables * n_nodes^n_dims * n_elements
    n_elements = nelements(integrator.p.solver, integrator.p.cache)
    resize!(integrator.p, n_elements)
end

function Base.resize!(semi::AbstractSemidiscretization, new_size)
    resize!(semi, semi.solver.volume_integral, new_size)
end

Base.resize!(semi, volume_integral::AbstractVolumeIntegral, new_size) = nothing

function Base.resize!(semi, volume_integral::VolumeIntegralSubcellLimiting, new_size)
    # Resize container antidiffusive_fluxes
    resize!(semi.cache.antidiffusive_fluxes, new_size)

    # Resize container subcell_limiter_coefficients
    @unpack limiter = volume_integral
    resize!(limiter.cache.subcell_limiter_coefficients, new_size)
end
end # @muladd
