# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    #! format: noindent
    
    # Abstract base type for time integration schemes of explicit strong stability-preserving (SSP)
    # Runge-Kutta (RK) methods. They are high-order time discretizations that guarantee the TVD property.
    abstract type SimpleAlgorithmIMEX end
    
    """
        SimpleIMEX(; stage_callbacks=())
    
        Pareschi - Russo IMEX Explicit Implicit IMEX-SSP2(3,3,2) Stiffly Accurate Scheme
    
    ## References
    
    - missing
    
    !!! warning "Experimental implementation"
        This is an experimental feature and may change in future releases.
    """
    struct SimpleIMEX{StageCallbacks} <: SimpleAlgorithmIMEX
       
       beta::Matrix{Float64}
       alfa::Matrix{Float64}
       gamma::Matrix{Float64}
       d::SVector{5,Float64}

       rkstages::Int64
       
       stage_callbacks::StageCallbacks

    
        function SimpleIMEX(; stage_callbacks = ())
            rkstages = 5
            beta = zeros(rkstages,rkstages)
            alfa = zeros(rkstages,rkstages)
            gamma = zeros(rkstages,rkstages) 
            d = zeros(rkstages,1)
            beta[2,1]=  0.38758444641450318
            beta[3,1]=  -2.5318448354142823E-002
            beta[3,2]=  0.38668943087310403
            beta[4,1]=  0.20899983523553325
            beta[4,2]= -0.45856648476371231
            beta[4,3]=  0.43423187573425748
            beta[5,1]= -0.10048822195663100
            beta[5,2]= -0.46186171956333327
            beta[5,3]=  0.83045062122462809
             beta[5,4]=  0.27014914900250392
             
             alfa[3,2]=  0.52349249922385610
             alfa[4,2]=   1.1683374366893629
             alfa[4,3]= -0.75762080241712637
             alfa[5,2]=  -3.6477233846797109E-002
             alfa[5,3]=  0.56936148730740477
             alfa[5,4]=  0.47746263002599681
             
             gamma[3,2]=  0.13145089796226542
             gamma[4,2]= -0.36855857648747881
             gamma[4,3]=  0.33159232636600550
             gamma[5,2]=  -6.5767130537473045E-002
             gamma[5,3]=   4.0591093109036858E-002
             gamma[5,4]=   6.4902111640806712E-002
             
             d2 = beta[2,1]
             d3 = beta[3,1] + beta[3,2]
             d4 = beta[4,1] + beta[4,2] + beta[4,3]
             d5 = beta[5,1] + beta[5,2] + beta[5,3] + beta[5,4]
             d = SVector(0.0,d2,d3,d4,d5)
            new{typeof(stage_callbacks)}(beta, alfa, gamma, d, rkstages, stage_callbacks)
        end
    end
    
    # This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
    mutable struct SimpleIntegratorIMEXOptions{Callback, TStops}
        callback::Callback # callbacks; used in Trixi
        adaptive::Bool # whether the algorithm is adaptive; ignored
        dtmax::Float64 # ignored
        maxiters::Int # maximal number of time steps
        tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    end
    
    function SimpleIntegratorIMEXOptions(callback, tspan; maxiters = typemax(Int), kwargs...)
        tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
        # We add last(tspan) to make sure that the time integration stops at the end time
        push!(tstops_internal, last(tspan))
        # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
        # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
        push!(tstops_internal, 2 * last(tspan))
        SimpleIntegratorIMEXOptions{typeof(callback), typeof(tstops_internal)}(callback,
                                                                              false, Inf,
                                                                              maxiters,
                                                                              tstops_internal)
    end
    
    # This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
    # This implements the interface components described at
    # https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
    # which are used in Trixi.
    mutable struct SimpleIntegratorIMEX{RealT <: Real, uType, Params, Sol, F1, F2, Alg,
                                       SimpleIntegratorIMEXOptions}
        u::uType
        u_tmp::uType
        Zn0::uType
        dZn::uType
        du::uType
        r0::uType
        t::RealT
        tdir::RealT
        dt::RealT # current time step
        dtcache::RealT # manually set time step
        iter::Int # current number of time steps (iteration)
        p::Params # will be the semidiscretization from Trixi
        sol::Sol # faked
        f1::F1
        f2::F2
        alg::Alg
        opts::SimpleIntegratorIMEXOptions
        finalstep::Bool # added for convenience
        dtchangeable::Bool
        force_stepfail::Bool
    end
    
    """
        add_tstop!(integrator::SimpleIntegratorSSP, t)
    Add a time stop during the time integration process.
    This function is called after the periodic SaveSolutionCallback to specify the next stop to save the solution.
    """
    function add_tstop!(integrator::SimpleIntegratorIMEX, t)
        integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
            error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
        # We need to remove the first entry of tstops when a new entry is added.
        # Otherwise, the simulation gets stuck at the previous tstop and dt is adjusted to zero.
        if length(integrator.opts.tstops) > 1
            pop!(integrator.opts.tstops)
        end
        push!(integrator.opts.tstops, integrator.tdir * t)
    end
    
    has_tstop(integrator::SimpleIntegratorIMEX) = !isempty(integrator.opts.tstops)
    first_tstop(integrator::SimpleIntegratorIMEX) = first(integrator.opts.tstops)
    
    # Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
    function Base.getproperty(integrator::SimpleIntegratorIMEX, field::Symbol)
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
    
    !!! warning "Experimental implementation"
        This is an experimental feature and may change in future releases.
    """
    function solve(ode::ODEProblem, alg = SimpleAlgorithmIMEX()::SimpleAlgorithmIMEX;
                   dt, callback = nothing, kwargs...)
        u = copy(ode.u0)
        u_tmp = similar(u)
        Zn0 = similar(u)
        dZn = similar(u)
        du = similar(u)
        r0 = similar(u)
        t = first(ode.tspan)
        tdir = sign(ode.tspan[end] - ode.tspan[1])
        iter = 0
        integrator = SimpleIntegratorIMEX(u, u_tmp, Zn0, dZn, du, r0, t, tdir, dt, dt, iter, ode.p,
                                         (prob = ode,), ode.f.f1, ode.f.f2, alg,
                                         SimpleIntegratorIMEXOptions(callback, ode.tspan;
                                                                    kwargs...),
                                         false, true, false)
    
        # resize container
        resize!(integrator.p, nelements(integrator.p.solver1, integrator.p.cache1))
    
        # initialize callbacks
        if callback isa CallbackSet
            foreach(callback.continuous_callbacks) do cb
                error("unsupported")
            end
            foreach(callback.discrete_callbacks) do cb
           #   cb.initialize(cb, integrator.u, integrator.t, integrator)
            end
        elseif !isnothing(callback)
            error("unsupported")
        end
    
        for stage_callback in alg.stage_callbacks
            init_callback(stage_callback, integrator.p)
        end
    
        solve!(integrator)
    end
    
    function solve!(integrator::SimpleIntegratorIMEX)
        @unpack prob = integrator.sol
        @unpack alg = integrator
        t_end = last(prob.tspan)
        callbacks = integrator.opts.callback
        semi = integrator.p
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        integrator.finalstep = false
        a = equations.a
        b = equations.b
        dts =  integrator.dt*a/b
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

                    t_stage = integrator.t + integrator.dt
                    
                    Yn = [ zeros(size(integrator.u)) for _ in 1:(alg.rkstages)]
                    Sdu = [ zeros(size(integrator.u)) for _ in 1:(alg.rkstages)]
                    for stage in 1:alg.rkstages
                        integrator.Zn0 .= integrator.u
                        
                        InitialConditionMIS!(integrator.Zn0,Yn,integrator.u,alg.alfa,stage)
                        
                        integrator.dZn .= 0
                       
                        PreparationODEMIS!(integrator.dZn, Yn, Sdu, integrator.u, alg.d, alg.gamma, alg.beta, integrator.dt, stage)

                        if stage == 1
                            Yn[stage] .= integrator.u
                        else
                            solveODEMIS!(Yn, integrator.dZn, integrator.Zn0, dts, alg.d[stage]*integrator.dt, integrator, prob, stage)
                        end

                        integrator.u_tmp .= Yn[stage]
                        integrator.f1(integrator.du, integrator.u_tmp, prob.p, integrator.t + integrator.dt)
                        Sdu[stage] .= integrator.du

                        # println("Print Yn")
                        # debuggingprintstage(Yn, mesh, equations, solver, cache,stage)
                    end
                    
                 #   if integrator.iter == 1
                 #   throw(error)
                 #   end
                    integrator.u .= Yn[end]
            integrator.iter += 1
            integrator.t += integrator.dt
    
            # handle callbacks
            if callbacks isa CallbackSet
                foreach(callbacks.discrete_callbacks) do cb
                    if cb.condition(integrator.u, integrator.t, integrator)
          #              cb.affect!(integrator)
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


    function debuggingprint(u, mesh, equations, solver, cache)
        u_wrap = wrap_array(u, mesh, equations, solver, cache)
        @show u_wrap[1,:,:]
        @show u_wrap[2,:,:]
        return nothing
    end

    function debuggingprintstage(u, mesh, equations, solver, cache,stage)
        u_stage = copy(u[stage])
        u_wrap = wrap_array(u_stage, mesh, equations, solver, cache)
        @show u_wrap[1,:,:]
        @show u_wrap[2,:,:]
        return nothing
    end

    function InitialConditionMIS!(Zn0,Yn,u,alfa,stage)

        for j in 1:(stage-1)
            @threaded for i in eachindex(Zn0)
                Zn0[i] = Zn0[i] + alfa[stage,j]*(Yn[j][i]- u[i]) 
            end
        end

    end

    function PreparationODEMIS!(dZn,Yn,Sdu,u,d,gamma,beta,dtL,stage)
        for j in 1:(stage-1)
            @threaded for i in eachindex(dZn)
              dZn[i] = dZn[i] + 1/d[stage]*(1/dtL*gamma[stage,j]*(Yn[j][i]-u[i]) + beta[stage,j]*Sdu[j][i]) 
            end
        end
    end

    function solveODEMIS!(Yn, dZn, Zn0, dt, dtL, integrator, prob, stage)    

        yn = copy(Zn0)
        y = zeros(size(yn))
        du = [ zeros(size(dZn)) for _ in 1:4]

        numit = round(dtL/dt)
        dtloc = dtL/numit
        A = zeros(4,4)
        b = zeros(4,1)
        A[2,1] = 0.5
        A[3,2] = 0.5
        A[4,3] = 1
        b[1] = 1/6
        b[2] = 1/3
        b[3] = 1/3
        b[4] = 1/6

        for ii in 1:numit

            for s in 1:4
                y .= yn
                for i in 1:(s-1)
                    y .= y + dtloc*A[s,i]*du[i]
                end
                integrator.u_tmp .= y
                integrator.f2(integrator.du, integrator.u_tmp, prob.p, integrator.t + integrator.dt)
                du[s] .= integrator.du + dZn
                if s == 4
                    y .= yn
                    for i in 1:4
                        y .= y + dtloc*b[i]*du[i]
                    end
                end     
            end
            yn .= y

        end
        Yn[stage] .= yn
    end
    
    # get a cache where the RHS can be stored
    get_du(integrator::SimpleIntegratorIMEX) = integrator.du
    get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.r0,)
    get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.Zn0,)
    get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.dZn,)
    
    # some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
    u_modified!(integrator::SimpleIntegratorIMEX, ::Bool) = false
    
    # used by adaptive timestepping algorithms in DiffEq
    function set_proposed_dt!(integrator::SimpleIntegratorIMEX, dt)
        (integrator.dt = dt; integrator.dtcache = dt)
    end
    
    # used by adaptive timestepping algorithms in DiffEq
    function get_proposed_dt(integrator::SimpleIntegratorIMEX)
        return ifelse(integrator.opts.adaptive, integrator.dt, integrator.dtcache)
    end
    
    # stop the time integration
    function terminate!(integrator::SimpleIntegratorIMEX)
        integrator.finalstep = true
    end
    
    """
        modify_dt_for_tstops!(integrator::SimpleIntegratorSSP)
    Modify the time-step size to match the time stops specified in integrator.opts.tstops.
    To avoid adding OrdinaryDiffEq to Trixi's dependencies, this routine is a copy of
    https://github.com/SciML/OrdinaryDiffEq.jl/blob/d76335281c540ee5a6d1bd8bb634713e004f62ee/src/integrators/integrator_utils.jl#L38-L54
    """
    function modify_dt_for_tstops!(integrator::SimpleIntegratorIMEX)
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
    function Base.resize!(integrator::SimpleIntegratorIMEX, new_size)
        resize!(integrator.u, new_size)
        resize!(integrator.du, new_size)
        resize!(integrator.r0, new_size)
    
        # Resize container
        # new_size = n_variables * n_nodes^n_dims * n_elements
        n_elements = nelements(integrator.p.solver, integrator.p.cache)
        resize!(integrator.p, n_elements)
    end
    
    function Base.resize!(semi::AbstractSemidiscretization, new_size)
        resize!(semi, semi.solver1.volume_integral, new_size)
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
    