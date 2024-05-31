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
    A1::Matrix{Float64}
    A2::Matrix{Float64}
    b::SVector{3, Float64}
    c1::SVector{3, Float64}
    c2::SVector{3, Float64}
    stage_callbacks::StageCallbacks

    function SimpleIMEX(; stage_callbacks = ())
        # Mathematically speaking, it is not necessary for the algorithm to split the factors
        # into numerator and denominator. Otherwise, however, rounding errors of the order of
        # the machine accuracy will occur, which will add up over time and thus endanger the
        # conservation of the simulation.
        # See also https://github.com/trixi-framework/Trixi.jl/pull/1640.
        A1 = zeros(3,3)
        A2 = zeros(3,3)
        A1[2,1] = 0.5
        A1[3,1] = 0.5
        A1[3,2] = 0.5

        A2[1,1] = 1/4
        A2[2,2] = 1/4
        A2[3,1] = 1/3
        A2[3,2] = 1/3
        A2[3,3] = 1/3

        b = SVector(1/3, 1/3, 1/3) 
        c1 = SVector(0.0, 1/2, 1)
        c2 = SVector(1/4, 1/4, 1)

        # Butcher tableau
        #   c |       a
        #   0 |
        #   1 |   1
        # 1/2 | 1/4  1/4
        # --------------------
        #   b | 1/6  1/6  2/3

        new{typeof(stage_callbacks)}(A1, A2, b, c1,c2,
                                     stage_callbacks)
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
mutable struct SimpleIntegratorIMEX{RealT <: Real, uType, Params, Sol, F, Alg,
                                   SimpleIntegratorIMEXOptions}
    u::uType
    u1::uType
    u2::uType
    u3::uType
    fu1::uType
    fu2::uType
    fu3::uType
    du1::uType
    du2::uType
    du3::uType
    du::uType
    u_tmp1::uType
    u_tmp2::uType
    u_tmp3::uType
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
    du1 = similar(u)
    du2 = similar(u)
    du3 = similar(u)
    fu1 = similar(u)
    fu2 = similar(u)
    fu3 = similar(u)
    du = similar(u)
    u1 = similar(u)
    u2 = similar(u)
    u3 = similar(u)
    u_tmp1 = similar(u)
    u_tmp2 = similar(u)
    u_tmp3 = similar(u)
    r0 = similar(u)
    t = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0
    integrator = SimpleIntegratorIMEX(u,u1,u2,u3,fu1,fu2,fu3,du1,du2,du3,du,u_tmp1,u_tmp2,u_tmp3, r0, t, tdir, dt, dt, iter, ode.p,
                                     (prob = ode,), ode.f, alg,
                                     SimpleIntegratorIMEXOptions(callback, ode.tspan;
                                                                kwargs...),
                                     false, true, false)

    # resize container
    resize!(integrator.p, nelements(integrator.p.solver, integrator.p.cache))

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            error("unsupported")
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
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
    relaxation_rate = equations.eps_relaxation
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
            integrator.u1 .= 0
            integrator.u2 .= 0
            integrator.u3 .= 0
            integrator.du1 .= 0
            integrator.du2 .= 0
            integrator.du3 .= 0
            integrator.fu1 .= 0
            integrator.fu2 .= 0
            integrator.fu3 .= 0
            integrator.u_tmp1 .= 0
            integrator.u_tmp2 .= 0
            integrator.u_tmp3 .= 0
                
                #Ui = Un - dt sum A1_ij partial rhs(Uj) - dt/epsilon sum A2_ij (Vj - f(Uj))
                #Un+1 = Un - dt sum bj partial rhs(Uj) - dt/epsilon sum bj (Vj - f(Uj))
                
                t_stage = integrator.t + integrator.dt * alg.c1[1]
                
                #   Stage 1
                #Ui = (ui;vi)
                #u1 = un
                #v1 = vn - -dt/epsilon A2_{11} (v1 - f(u1)) 
              @. integrator.u1 .= integrator.u  #u1 = v1 = un
                
              @. integrator.fu1 .= integrator.u1
                 wrap_and_perform_projection!(integrator.fu1,integrator.dt,mesh,equations,solver,cache)  # compute f(u1)

              @. integrator.u1 = integrator.u1 + integrator.dt/relaxation_rate*alg.A2[1,1]*integrator.fu1 # v1 = vn + dt/eps*A2_11 f(u1)
                
                divide_relaxed_var!(integrator.u1,integrator.dt,semi,solver,cache,alg.A2[1,1],equations,mesh)  # v1 = (vn + dt/eps*A2_11 f(u1))/(1 + dt/eps A2_11)
               
                integrator.f(integrator.du1, integrator.u1, integrator.p, t_stage) # compute RHS(u1,v1)

                # Stage 2        
                #u2 = un - dt A1_{21} rhs(u1,v1) 
                #v2 = vn - dt A1_{21} rhs(u1,v1) - dt/epsilon A2_{21} (v1 - f(u1)) -dt/epsilon A2_{22} (v2 - f(u2)) 
                
                t_stage = integrator.t + integrator.dt * alg.c1[2]
           
                @. integrator.u2 = integrator.u + integrator.dt*alg.A1[2,1]*integrator.du1 # u2 = v2 = Un - dt*A1_22 RHS(U1)

                @. integrator.fu2 = integrator.u2
                wrap_and_perform_projection!(integrator.fu2,integrator.dt,mesh,equations,solver,cache) # compute f(u2) and setting the source term values to 0 for the cons variables
                
                @. integrator.u_tmp1 = integrator.u1
                set_cons_var_to_zero!(integrator.u_tmp1,semi,solver,cache,equations,mesh) # computing v1 by setting cons variables to 0

                # v2 = vn - dt/eps*A2_21*(v1-f(u1)) + dt/eps*A2_22*f(u2)
                @. integrator.u2 = integrator.u2 - integrator.dt/relaxation_rate*alg.A2[2,1]*(integrator.u_tmp1 - integrator.fu1) + integrator.dt*alg.A2[2,2]/relaxation_rate*integrator.fu2
               
                divide_relaxed_var!(integrator.u2,integrator.dt,semi,solver,cache,alg.A2[2,2],equations,mesh) # v2 = (vn - dt/eps*A2_21*(v1-f(u1)) + dt/eps*A2_22*f(u2) ) ( 1+dt/eps A2_22)

                integrator.f(integrator.du2, integrator.u2, integrator.p, t_stage)
               
                # Stage 3
                #u3 = un - dt A1_{31} rhs(u1,v1) - dt A1_{32} rhs(u2,v2)
                #v3 = vn - dt A1_{31} rhs(u1,v1) - dt A1_{32} rhs(u2,v2) - dt/epsilon A2_{31} (v1 - f(u1)) -dt/epsilon A2_{32} (v2 - f(u2)) -dt/epsilon A2_{33} (v3 - f(u3))
                @. integrator.u3 = integrator.u + integrator.dt*alg.A1[3,1]*integrator.du1 + integrator.dt*alg.A1[3,2]*integrator.du2

                @. integrator.fu3 = integrator.u3
                wrap_and_perform_projection!(integrator.fu3,integrator.dt,mesh,equations,solver,cache)

                #  @. integrator.u_tmp1 = integrator.u1
                #  set_cons_var_to_zero!(integrator.u_tmp1,semi,solver,cache)

                @. integrator.u_tmp2 = integrator.u2
                set_cons_var_to_zero!(integrator.u_tmp2,semi,solver,cache,equations,mesh)

                @. integrator.u3 = integrator.u3 - integrator.dt/relaxation_rate*alg.A2[3,1]*(integrator.u_tmp1 - integrator.fu1) - integrator.dt/relaxation_rate*alg.A2[3,2]*(integrator.u_tmp2 - integrator.fu2) + integrator.dt*alg.A2[3,3]/relaxation_rate*integrator.fu3
                
                divide_relaxed_var!(integrator.u3,integrator.dt,semi,solver,cache,alg.A2[3,3],equations,mesh) 
                
                integrator.f(integrator.du3, integrator.u3, integrator.p, t_stage)
                
                # Final Stage
                @. integrator.u = integrator.u + integrator.dt*alg.b[1]*integrator.du1 + integrator.dt*alg.b[2]*integrator.du2 + integrator.dt*alg.b[3]*integrator.du3
                
                # Already done that for u_tmp1 and u_tmp2, such that they are v1 = u_tmp1 and v2 = u_tmp2
                # integrator.u_tmp1 .= integrator.u1              
                # integrator.u_tmp2 .= integrator.u2
                @. integrator.u_tmp3 = integrator.u3
                # set_cons_var_to_zero!(integrator.u_tmp1,semi,solver,cache)
                # set_cons_var_to_zero!(integrator.u_tmp2,semi,solver,cache)
                set_cons_var_to_zero!(integrator.u_tmp3,semi,solver,cache,equations,mesh)

                @. integrator.u = integrator.u - integrator.dt/relaxation_rate*alg.b[1]*(integrator.u_tmp1 - integrator.fu1) - integrator.dt/relaxation_rate*alg.b[2]*(integrator.u_tmp2 - integrator.fu2) - integrator.dt*alg.b[3]/relaxation_rate*(integrator.u_tmp3 - integrator.fu3)
                # End Stages   
          #  for stage_callback in alg.stage_callbacks
          #      stage_callback(integrator.u, integrator, 1)
          #  end

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

function divide_relaxed_var!(u,dt,semi,solver,cache,aii,equations,mesh)
    u_wrap = Trixi.wrap_array(u,semi)
    cycle_divide!(u_wrap,dt,semi,solver,cache,aii,equations,mesh)
    return nothing
end

function cycle_divide!(u,dt,semi,solver,cache,aii,equations,mesh::TreeMesh2D)
    @unpack inverse_jacobian = cache.elements
                nvars_base = nvariables(equations.equations_base)
         relaxation_rate = equations.eps_relaxation
                for element in eachelement(solver,cache)
                    factor = inverse_jacobian[element]
                    for j in eachnode(solver),i in eachnode(solver)
                        for var in (nvars_base+1):(nvars_base*3)
                        u[var,i,j,element] = u[var,i,j,element]/(1.0+factor*dt/relaxation_rate*aii)    
                        end
                    end
                end

    return nothing
end


function cycle_divide!(u,dt,semi,solver,cache,aii,equations,mesh::TreeMesh1D)
    @unpack inverse_jacobian = cache.elements
                nvars_base = nvariables(equations.equations_base)
         relaxation_rate = equations.eps_relaxation
                for element in eachelement(solver,cache)
                    factor = inverse_jacobian[element]
                    for i in eachnode(solver)
                        for var in (nvars_base+1):(nvars_base*2)
                        u[var,i,element] = u[var,i,element]/(1.0+factor*dt/relaxation_rate*aii)    
                        end
                    end
                end

    return nothing
end

function wrap_and_perform_projection!(u,dt,mesh,equations,solver,cache)

                u_wrap = wrap_array(u, mesh, equations, solver, cache)
                perform_projection_sourceterm!(u_wrap,dt,mesh,equations,solver,cache)
    
    return nothing
end


function set_cons_var_to_zero!(u,semi,solver,cache,equations, mesh::TreeMesh2D)
                u_wrap = Trixi.wrap_array(u,semi)
                nvars_base = nvariables(equations.equations_base)
    @unpack inverse_jacobian = cache.elements
                for element in eachelement(solver, cache)    
                    factor = inverse_jacobian[element]
                    for j in eachnode(solver), i in eachnode(solver)
                        for var in 1:nvars_base
                           u_wrap[var,i,j,element] = 0.0
                        end
                        for var in (nvars_base+1):(nvars_base*3)
                           u_wrap[var,i,j,element] *= factor 
                        end
                    end
                end
    return nothing
end


function set_cons_var_to_zero!(u,semi,solver,cache,equations, mesh::TreeMesh1D)
                u_wrap = Trixi.wrap_array(u,semi)
                nvars_base = nvariables(equations.equations_base)
    @unpack inverse_jacobian = cache.elements
                for element in eachelement(solver, cache)    
                    factor = inverse_jacobian[element]
                    for i in eachnode(solver)
                        for var in 1:nvars_base
                           u_wrap[var,i,element] = 0.0
                        end
                        for var in (nvars_base+1):(nvars_base*2)
                           u_wrap[var,i,element] *= factor 
                        end
                    end
                end
    return nothing
end

function perform_projection_sourceterm!(u, dt, mesh::TreeMesh1D, equations::JinXinEquations, dg, cache)

    # relaxation parameter
    eps = equations.eps_relaxation
    dt_ = dt
    eq_relax = equations.equations_base

    @unpack inverse_jacobian = cache.elements
    # prepare local storage for projection
    @unpack interpolate_N_to_M, project_M_to_N, filter_modal_to_N = dg.basis
    nnodes_,nnodes_projection = size(project_M_to_N)
    nVars = nvariables(eq_relax)
    RealT = real(dg)
    u_N = zeros(RealT, nVars, nnodes_)
    w_N = zeros(RealT, nVars, nnodes_)
    f_N = zeros(RealT, nVars, nnodes_)
    g_N = zeros(RealT, nVars, nnodes_)
    u_M = zeros(RealT, nVars, nnodes_projection)
    w_M_raw = zeros(RealT, nVars, nnodes_projection)
    w_M = zeros(RealT, nVars, nnodes_projection)
    f_M = zeros(RealT, nVars, nnodes_projection)
    g_M = zeros(RealT, nVars, nnodes_projection)

    tmp_MxM = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
    tmp_MxN = zeros(RealT, nVars, nnodes_projection, nnodes_)
    tmp_NxM = zeros(RealT, nVars, nnodes_, nnodes_projection)

#@threaded for element in eachelement(dg, cache)
for element in eachelement(dg, cache)

                    factor = inverse_jacobian[element]
# get element u_N
for i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, element)
    for v in eachvariable(eq_relax)
        u_N[v,i] = u_node[v]
    end
end
# bring elemtn u_N to grid (M+1)x(M+1)

multiply_dimensionwise!(u_M,interpolate_N_to_M,u_N)

# compute nodal values of entropy variables w on the M grid
for i in 1:nnodes_projection
    u_cons = get_node_vars(u_M, eq_relax, dg, i)
    w_ij   = cons2entropy(u_cons,eq_relax) 
    set_node_vars!(w_M_raw,w_ij,eq_relax,dg,i)
end
# compute projection of w with M values down to N
multiply_dimensionwise!(w_M,filter_modal_to_N,w_M_raw)

#multiply_dimensionwise!(w_N,project_M_to_N,w_M)
#multiply_dimensionwise!(w_M,interpolate_N_to_M,w_N)


# compute nodal values of conservative f,g on the M grid
for i in 1:nnodes_projection
    w_i = get_node_vars(w_M, eq_relax, dg, i)
    u_cons = entropy2cons(w_i, eq_relax)
    f_cons = flux(u_cons,1,eq_relax)
    set_node_vars!(f_M,f_cons,eq_relax,dg,i)
end
# compute projection of f with M values down to N, same for g
multiply_dimensionwise!(f_N,project_M_to_N,f_M)
#@assert nnodes_projection == nnodes(dg) 
#for j in 1:nnodes_projection, i in 1:nnodes_projection
#    u_cons = get_node_vars(u_N, eq_relax, dg, i, j)
#    f_cons = flux(u_cons,1,eq_relax)
#    set_node_vars!(f_N,f_cons,eq_relax,dg,i,j)
#    g_cons = flux(u_cons,2,eq_relax)
#    set_node_vars!(g_N,g_cons,eq_relax,dg,i,j)
#end

    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)
        # compute compressible Euler fluxes
        vu = get_node_vars(f_N,eq_relax,dg,i)
        u_base = get_block_components2(u_node, 1, equations)
        new_u = factor*SVector(zero(u_node)..., vu...)
        set_node_vars!(u, new_u, equations, dg, i, element)
    end
end
return nothing
end

function perform_projection_sourceterm!(u, dt, mesh::TreeMesh2D, equations::JinXinEquations, dg, cache)

    # relaxation parameter
    eps = equations.eps_relaxation
    dt_ = dt
    eq_relax = equations.equations_base

    @unpack inverse_jacobian = cache.elements
    # prepare local storage for projection
    @unpack interpolate_N_to_M, project_M_to_N, filter_modal_to_N = dg.basis
    nnodes_,nnodes_projection = size(project_M_to_N)
    nVars = nvariables(eq_relax)
    RealT = real(dg)
    u_N = zeros(RealT, nVars, nnodes_, nnodes_)
    w_N = zeros(RealT, nVars, nnodes_, nnodes_)
    f_N = zeros(RealT, nVars, nnodes_, nnodes_)
    g_N = zeros(RealT, nVars, nnodes_, nnodes_)
    u_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
    w_M_raw = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
    w_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
    f_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
    g_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)

    tmp_MxM = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
    tmp_MxN = zeros(RealT, nVars, nnodes_projection, nnodes_)
    tmp_NxM = zeros(RealT, nVars, nnodes_, nnodes_projection)

#@threaded for element in eachelement(dg, cache)
for element in eachelement(dg, cache)

                    factor = inverse_jacobian[element]
# get element u_N
for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)
    for v in eachvariable(eq_relax)
        u_N[v,i,j] = u_node[v]
    end
end
# bring elemtn u_N to grid (M+1)x(M+1)
multiply_dimensionwise!(u_M,interpolate_N_to_M,u_N,tmp_MxN)

# compute nodal values of entropy variables w on the M grid
for j in 1:nnodes_projection, i in 1:nnodes_projection
    u_cons = get_node_vars(u_M, eq_relax, dg, i, j)
    w_ij   = cons2entropy(u_cons,eq_relax) 
    set_node_vars!(w_M_raw,w_ij,eq_relax,dg,i,j)
end
# compute projection of w with M values down to N
multiply_dimensionwise!(w_M,filter_modal_to_N,w_M_raw,tmp_MxM)

#multiply_dimensionwise!(w_N,project_M_to_N,w_M)
#multiply_dimensionwise!(w_M,interpolate_N_to_M,w_N)


# compute nodal values of conservative f,g on the M grid
for j in 1:nnodes_projection, i in 1:nnodes_projection
    w_ij = get_node_vars(w_M, eq_relax, dg, i, j)
    u_cons = entropy2cons(w_ij, eq_relax)
    f_cons = flux(u_cons,1,eq_relax)
    set_node_vars!(f_M,f_cons,eq_relax,dg,i,j)
    g_cons = flux(u_cons,2,eq_relax)
    set_node_vars!(g_M,g_cons,eq_relax,dg,i,j)
end
# compute projection of f with M values down to N, same for g
multiply_dimensionwise!(f_N,project_M_to_N,f_M,tmp_NxM)
multiply_dimensionwise!(g_N,project_M_to_N,g_M,tmp_NxM)
#@assert nnodes_projection == nnodes(dg) 
#for j in 1:nnodes_projection, i in 1:nnodes_projection
#    u_cons = get_node_vars(u_N, eq_relax, dg, i, j)
#    f_cons = flux(u_cons,1,eq_relax)
#    set_node_vars!(f_N,f_cons,eq_relax,dg,i,j)
#    g_cons = flux(u_cons,2,eq_relax)
#    set_node_vars!(g_N,g_cons,eq_relax,dg,i,j)
#end

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        # compute compressible Euler fluxes
        vu = get_node_vars(f_N,eq_relax,dg,i,j)
        wu = get_node_vars(g_N,eq_relax,dg,i,j)
        u_base = get_block_components2(u_node, 1, equations)
        new_u = factor*SVector(zero(u_node)..., vu..., wu...)
        set_node_vars!(u, new_u, equations, dg, i, j, element)
    end
end
return nothing
end

function get_block_components2(u, n, equations::JinXinEquations)
    nvars_base = nvariables(equations.equations_base)
    return SVector(ntuple(i -> u[i + (n - 1) * nvars_base], Val(nvars_base)))
end


# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegratorIMEX) = integrator.du
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.r0,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.fu1,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.fu2,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.fu3,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.du1,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.du2,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.du3,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.u1,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.u2,)
get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.u3,)

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
