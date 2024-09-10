# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    #! format: noindent

    # Abstract base type for time integration schemes of storage class `3S*`
    abstract type SimpleAlgorithm3Sstar end

    """
        HypDiffN3Erk3Sstar52()

    Five stage, second-order accurate explicit Runge-Kutta scheme with stability region optimized for
    the hyperbolic diffusion equation with LLF flux and polynomials of degree polydeg=3.
    """
    struct HypDiffN3Erk3Sstar52 <: SimpleAlgorithm3Sstar
        gamma1::SVector{5, Float64}
        gamma2::SVector{5, Float64}
        gamma3::SVector{5, Float64}
        beta::SVector{5, Float64}
        delta::SVector{5, Float64}
        c::SVector{5, Float64}

        function HypDiffN3Erk3Sstar52()
            gamma1 = SVector(
                0.0e+0, 5.2656474556752575e-1,
                1.0385212774098265e+0, 3.6859755007388034e-1,
                -6.3350615190506088e-1
            )
            gamma2 = SVector(
                1.0e+0, 4.1892580153419307e-1,
                -2.7595818152587825e-2, 9.1271323651988631e-2,
                6.8495995159465062e-1
            )
            gamma3 = SVector(
                0.0e+0, 0.0e+0,
                0.0e+0, 4.1301005663300466e-1,
                -5.4537881202277507e-3
            )
            beta = SVector(
                4.5158640252832094e-1, 7.5974836561844006e-1,
                3.7561630338850771e-1, 2.9356700007428856e-2,
                2.5205285143494666e-1
            )
            delta = SVector(
                1.0e+0, 1.3011720142005145e-1,
                2.6579275844515687e-1, 9.9687218193685878e-1,
                0.0e+0
            )
            c = SVector(
                0.0e+0, 4.5158640252832094e-1,
                1.0221535725056414e+0, 1.4280257701954349e+0,
                7.1581334196229851e-1
            )

            new(gamma1, gamma2, gamma3, beta, delta, c)
        end
    end

    """
        ParsaniKetchesonDeconinck3Sstar94()

    Parsani, Ketcheson, Deconinck (2013)
      Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
    [DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
    """
    struct ParsaniKetchesonDeconinck3Sstar94 <: SimpleAlgorithm3Sstar
        gamma1::SVector{9, Float64}
        gamma2::SVector{9, Float64}
        gamma3::SVector{9, Float64}
        beta::SVector{9, Float64}
        delta::SVector{9, Float64}
        c::SVector{9, Float64}

        function ParsaniKetchesonDeconinck3Sstar94()
            gamma1 = SVector(
                0.0e+0, -4.6556413837561301e+0,
                -7.7202649689034453e-1, -4.0244202720632174e+0,
                -2.1296873883702272e-2, -2.4350219407769953e+0,
                1.9856336960249132e-2, -2.8107894116913812e-1,
                1.68943543736779e-1
            )
            gamma2 = SVector(
                1.0e+0, 2.4992627683300688e+0,
                5.8668202764174726e-1, 1.2051419816240785e+0,
                3.4747937498564541e-1, 1.3213458736302766e+0,
                3.1196363453264964e-1, 4.3514189245414447e-1,
                2.3596980658341213e-1
            )
            gamma3 = SVector(
                0.0e+0, 0.0e+0,
                0.0e+0, 7.6209857891449362e-1,
                -1.981181783296552e-1, -6.2289587091629484e-1,
                -3.7522475499063573e-1, -3.3554373281046146e-1,
                -4.5609629702116454e-2
            )
            beta = SVector(
                2.8363432481011769e-1, 9.7364980747486463e-1,
                3.3823592364196498e-1, -3.5849518935750763e-1,
                -4.1139587569859462e-3, 1.4279689871485013e+0,
                1.8084680519536503e-2, 1.6057708856060501e-1,
                2.9522267863254809e-1
            )
            delta = SVector(
                1.0e+0, 1.2629238731608268e+0,
                7.5749675232391733e-1, 5.1635907196195419e-1,
                -2.7463346616574083e-2, -4.3826743572318672e-1,
                1.2735870231839268e+0, -6.294738221773023e-1,
                0.0e+0
            )
            c = SVector(
                0.0e+0, 2.8363432481011769e-1,
                5.4840742446661772e-1, 3.6872298094969475e-1,
                -6.8061183026103156e-1, 3.5185265855105619e-1,
                1.6659419385562171e+0, 9.7152778807463247e-1,
                9.0515694340066954e-1
            )

            new(gamma1, gamma2, gamma3, beta, delta, c)
        end
    end

    """
        ParsaniKetchesonDeconinck3Sstar32()

    Parsani, Ketcheson, Deconinck (2013)
      Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
    [DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
    """
    struct ParsaniKetchesonDeconinck3Sstar32 <: SimpleAlgorithm3Sstar
        gamma1::SVector{3, Float64}
        gamma2::SVector{3, Float64}
        gamma3::SVector{3, Float64}
        beta::SVector{3, Float64}
        delta::SVector{3, Float64}
        c::SVector{3, Float64}

        function ParsaniKetchesonDeconinck3Sstar32()
            gamma1 = SVector(
                0.0e+0, -1.2664395576322218e-1,
                1.1426980685848858e+0
            )
            gamma2 = SVector(
                1.0e+0, 6.542778259940647e-1,
                -8.2869287683723744e-2
            )
            gamma3 = SVector(
                0.0e+0, 0.0e+0,
                0.0e+0
            )
            beta = SVector(
                7.2366074728360086e-1, 3.4217876502651023e-1,
                3.6640216242653251e-1
            )
            delta = SVector(
                1.0e+0, 7.2196567116037724e-1,
                0.0e+0
            )
            c = SVector(
                0.0e+0, 7.2366074728360086e-1,
                5.9236433182015646e-1
            )

            new(gamma1, gamma2, gamma3, beta, delta, c)
        end
    end

    mutable struct SimpleIntegrator3SstarOptions{Callback}
        callback::Callback # callbacks; used in Trixi.jl
        adaptive::Bool # whether the algorithm is adaptive; ignored
        dtmax::Float64 # ignored
        maxiters::Int # maximal number of time steps
        tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    end

    function SimpleIntegrator3SstarOptions(
            callback, tspan; maxiters = typemax(Int),
            kwargs...
        )
        SimpleIntegrator3SstarOptions{typeof(callback)}(
            callback, false, Inf, maxiters,
            [last(tspan)]
        )
    end

    mutable struct SimpleIntegrator3Sstar{
            RealT <: Real, uType, Params, Sol, F, Alg,
            SimpleIntegrator3SstarOptions,
        }
        u::uType #
        du::uType
        u_tmp1::uType
        u_tmp2::uType
        t::RealT
        dt::RealT # current time step
        dtcache::RealT # ignored
        iter::Int # current number of time step (iteration)
        p::Params # will be the semidiscretization from Trixi.jl
        sol::Sol # faked
        f::F
        alg::Alg
        opts::SimpleIntegrator3SstarOptions
        finalstep::Bool # added for convenience
    end

    # Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
    function Base.getproperty(integrator::SimpleIntegrator3Sstar, field::Symbol)
        if field === :stats
            return (naccept = getfield(integrator, :iter),)
        end
        # general fallback
        return getfield(integrator, field)
    end

    function init(
            ode::ODEProblem, alg::SimpleAlgorithm3Sstar;
            dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...
        )
        u = copy(ode.u0)
        du = similar(u)
        u_tmp1 = similar(u)
        u_tmp2 = similar(u)
        t = first(ode.tspan)
        iter = 0
        integrator = SimpleIntegrator3Sstar(
            u, du, u_tmp1, u_tmp2, t, dt, zero(dt), iter,
            ode.p,
            (prob = ode,), ode.f, alg,
            SimpleIntegrator3SstarOptions(
                callback,
                ode.tspan;
                kwargs...
            ), false
        )

        # initialize callbacks
        if callback isa CallbackSet
            foreach(callback.continuous_callbacks) do cb
                throw(ArgumentError("Continuous callbacks are unsupported with the 3 star time integration methods."))
            end
            foreach(callback.discrete_callbacks) do cb
                cb.initialize(cb, integrator.u, integrator.t, integrator)
            end
        end

        return integrator
    end

    # Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
    function solve(
            ode::ODEProblem, alg::SimpleAlgorithm3Sstar;
            dt, callback = nothing, kwargs...
        )
        integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

        # Start actual solve
        solve!(integrator)
    end

    function solve!(integrator::SimpleIntegrator3Sstar)
        @unpack prob = integrator.sol

        integrator.finalstep = false

        @trixi_timeit timer() "main loop" while !integrator.finalstep
            step!(integrator)
        end # "main loop" timer

        return TimeIntegratorSolution(
            (first(prob.tspan), integrator.t),
            (prob.u0, integrator.u),
            integrator.sol.prob
        )
    end

    function step!(integrator::SimpleIntegrator3Sstar)
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

        # one time step
        integrator.u_tmp1 .= zero(eltype(integrator.u_tmp1))
        integrator.u_tmp2 .= integrator.u
        for stage in eachindex(alg.c)
            t_stage = integrator.t + integrator.dt * alg.c[stage]
            prob.f(integrator.du, integrator.u, prob.p, t_stage)

            delta_stage = alg.delta[stage]
            gamma1_stage = alg.gamma1[stage]
            gamma2_stage = alg.gamma2[stage]
            gamma3_stage = alg.gamma3[stage]
            beta_stage_dt = alg.beta[stage] * integrator.dt
            @trixi_timeit timer() "Runge-Kutta step" begin
                @threaded for i in eachindex(integrator.u)
                    integrator.u_tmp1[i] += delta_stage * integrator.u[i]
                    integrator.u[i] = (
                        gamma1_stage * integrator.u[i] +
                            gamma2_stage * integrator.u_tmp1[i] +
                            gamma3_stage * integrator.u_tmp2[i] +
                            beta_stage_dt * integrator.du[i]
                    )
                end
            end
        end
        integrator.iter += 1
        integrator.t += integrator.dt

        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end

        # respect maximum number of iterations
        if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
            @warn "Interrupted. Larger maxiters is needed."
            terminate!(integrator)
        end
    end

    # get a cache where the RHS can be stored
    get_du(integrator::SimpleIntegrator3Sstar) = integrator.du
    function get_tmp_cache(integrator::SimpleIntegrator3Sstar)
        (integrator.u_tmp1, integrator.u_tmp2)
    end

    # some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
    u_modified!(integrator::SimpleIntegrator3Sstar, ::Bool) = false

    # used by adaptive timestepping algorithms in DiffEq
    function set_proposed_dt!(integrator::SimpleIntegrator3Sstar, dt)
        integrator.dt = dt
    end

    # Required e.g. for `glm_speed_callback`
    function get_proposed_dt(integrator::SimpleIntegrator3Sstar)
        return integrator.dt
    end

    # stop the time integration
    function terminate!(integrator::SimpleIntegrator3Sstar)
        integrator.finalstep = true
        empty!(integrator.opts.tstops)
    end

    # used for AMR
    function Base.resize!(integrator::SimpleIntegrator3Sstar, new_size)
        resize!(integrator.u, new_size)
        resize!(integrator.du, new_size)
        resize!(integrator.u_tmp1, new_size)
        resize!(integrator.u_tmp2, new_size)
    end
end # @muladd
