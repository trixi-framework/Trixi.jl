using OrdinaryDiffEq
using Trixi
using UnPack
using Plots

using LinearAlgebra
using SparseArrays
using RecursiveArrayTools
using EntropyStableEuler
using Setfield
using StartUpDG
using NodesAndModes

###############################################################################
# semidiscretization 

N = 3
K1D = 16
CFL = .1
FinalTime = .1

elem_type = Quad()

vol_quad = NodesAndModes.quad_nodes(elem_type,2*N)
rd = RefElemData(elem_type,N; quad_rule_vol = vol_quad)
# rd = RefElemData(elem_type,N)
rd = @set rd.Vf = droptol!(sparse(rd.Vf),1e-12) # make sparse interp matrix

VX,VY,EToV = uniform_mesh(elem_type,K1D)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

struct UnstructuredMesh{NDIMS,Tv,Ti}
    VXYZ::NTuple{NDIMS,Tv}
    EToV::Matrix{Ti}
end

function Base.show(io::IO, mesh::UnstructuredMesh{NDIMS,Tv,Ti}) where {NDIMS,Tv,Ti}
    @nospecialize mesh
    println("Unstructured mesh of dimension $NDIMS with $(size(EToV,1)) elements")
end

# Euler functions
@inline function f(U)
    ρ,ρu,ρv,E = U
    u = ρu./ρ
    v = ρv./ρ    
    p = pfun(EntropyStableEuler.Euler{2}(),U)
    
    fm1x = @. ρu*u + p
    fm2x = @. ρu*v     
    fEx = @. (E + p)*u

    fm1y = @. ρu*v
    fm2y = @. ρv*v + p    
    fEy = @. (E + p)*v    
    return SVector{4}(ρu,fm1x,fm2x,fEx),SVector{4}(ρv,fm1y,fm2y,fEy)
end
@inline fEC(uL,uR) = fS(EntropyStableEuler.Euler{2}(),uL,uR) 
@inline v_u(u) = v_ufun(EntropyStableEuler.Euler{2}(),u) # entropy vars in terms of cons vars
@inline u_v(v) = u_vfun(EntropyStableEuler.Euler{2}(),v) # cons vars in terms of entropy vars
@inline S(u) = Sfun(EntropyStableEuler.Euler{2}(),u) # entropy 

struct EulerProject2D{F1,F2,F3,F4,F5}
    f::F1
    fEC::F2
    v_u::F3
    u_v::F4
    S::F5
end
Trixi.ndims(equations::EulerProject2D) = 2
Trixi.nvariables(equations::EulerProject2D) = 4

function initial_condition(xyz,t,equations::EulerProject2D)
    x,y = xyz
    ρ = 1. + .25*exp(-25*(x^2+y^2))
    # ρ = 1 + .5*(abs(x) < .25 && abs(y) < .25)
    u = 0.10
    v = 0.20
    p = 2.0 #ρ^1.4
    return prim2cons((ρ,u,v,p),CompressibleEulerEquations2D(1.4)) # hacky
end

Base.real(rd::RefElemData) = Float64

function Trixi.create_cache(mesh::UnstructuredMesh, equations, rd::RefElemData, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    @unpack M,Dr,Ds,Vf,Vq,Pq = rd
    Qr = M*Dr
    Qs = M*Ds
    invMQTr = Matrix(-M\Qr')
    invMQTs = Matrix(-M\Qs')

    dfxdr = similar(md.x)
    dfxds = similar(md.x)
    dfydr = similar(md.x)
    dfyds = similar(md.x)
    lifted_flux = similar(md.x)
    interface_flux = similar(md.xf)
    cache = (;md,invMQTr,invMQTs,dfxdr,dfxds,dfydr,dfyds,lifted_flux,interface_flux)

    return cache
end

@inline function max_abs_speed_normal(UL, UR, normal, equations::CompressibleEulerEquations2D) 
    # Calculate primitive variables and speed of sound
    ρu_n_L = UL[2]*normal[1] + UL[3]*normal[2]
    ρu_n_R = UR[2]*normal[1] + UR[3]*normal[2]
    uL = (UL[1],ρu_n_L,UL[4])
    uR = (UR[1],ρu_n_R,UR[4])
    return max_abs_speed_naive(uL,uR,normal,CompressibleEulerEquations1D(1.4))
end
LF(uL,uR,normal) = Trixi.DissipationLocalLaxFriedrichs(max_abs_speed_normal)(uL,uR,normal,CompressibleEulerEquations2D(1.4))

function apply_DG_deriv!(dQ,fx,fy,fxf,fyf,fx_ec,fy_ec,diss,rd,cache)
    @unpack md = cache
    @unpack dfxdr,dfxds,dfydr,dfyds,lifted_flux,interface_flux = cache
    @unpack invMQTr,invMQTs = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = md
    @unpack LIFT = rd
    for i = 1:length(dQ)
        @. interface_flux = nxJ*fxf[i] + nyJ*fyf[i] - diss[i]*sJ
        mul!(lifted_flux,LIFT,interface_flux)
        mul!(dfxdr,invMQTr,fx[i])
        mul!(dfxds,invMQTs,fx[i])
        mul!(dfydr,invMQTr,fy[i])
        mul!(dfyds,invMQTs,fy[i])
        @. dQ[i] = -(rxJ*dfxdr + sxJ*dfxds + ryJ*dfydr + syJ*dfyds + lifted_flux)/J
    end
end

function Trixi.rhs!(du, u::VectorOfArray, t,
                    mesh::UnstructuredMesh, equations::EulerProject2D,
                    initial_condition, boundary_conditions, source_terms,
                    rd::RefElemData, cache)

    @unpack md = cache
    @unpack invMQTr,invMQTs = cache
    @unpack f,fEC,v_u,u_v,S = equations
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack Vq,Pq,Vf,wf,LIFT = rd

    # unpack VecOfArray storage for broadcasting
    Q = u.u 
    dQ = du.u

    # @inline flux_x(uL,uR) = flux_chandrashekar(uL,uR,1,CompressibleEulerEquations2D(1.4))
    # @inline flux_y(uL,uR) = flux_chandrashekar(uL,uR,2,CompressibleEulerEquations2D(1.4))

    # entropy projection    
    VU = (x->Pq*x).(v_u((x->Vq*x).(Q)))
    Uvol = u_v((x->Vq*x).(VU))
    Uf = u_v((x->Vf*x).(VU))

    # project volume flux
    fx,fy = map.(x->Pq*x,f(Uvol))

    # compute face values   
    UP = (u->u[mapP]).(Uf)
    fx_ec, fy_ec = fEC(UP,Uf)

    fxf = (x->Vf*x).(fx)
    fyf = (x->Vf*x).(fy)
    fxavg = (u->@. .5*(u+u[mapP])).(fxf)
    fyavg = (u->@. .5*(u+u[mapP])).(fyf)

    # correction
    Δfx = @. fx_ec - fxavg
    Δfy = @. fy_ec - fyavg
    Δv = v_u(UP) .- v_u(Uf) # can also compute from ṽ
    sign_Δv = map(x->sign.(x),Δv)
    Δfx_dot_signΔv = sum(map((x,y)->x.*y.*nxJ,sign_Δv,Δfx))
    Δfy_dot_signΔv = sum(map((x,y)->x.*y.*nyJ,sign_Δv,Δfy))
    ctmp = @. max(0,-Δfx_dot_signΔv) + max(0,-Δfy_dot_signΔv)
    dissipation = map(x->ctmp.*x,sign_Δv) # central dissipation
    
    # LF dissipation
    for i = 1:length(first(dissipation))
        normal = (nxJ[i]/sJ[i],nyJ[i]/sJ[i])
        LFi = LF(getindex.(UP,i),getindex.(Uf,i),normal)
        for fld = 1:length(dissipation)
            dissipation[fld][i] += LFi[fld]
        end
    end

    # function rhs_scalar!(du,fx,fy,fxf,fyf,diss) 
    #     du .= -(rxJ.*(invMQTr*fx) .+ sxJ.*(invMQTs*fx) .+ 
    #             ryJ.*(invMQTr*fy) .+ syJ.*(invMQTs*fy) .+
    #             LIFT*(@. nxJ*fxf + nyJ*fyf - diss*sJ))./J
    # end    
    # rhs_scalar!.(dQ,fx,fy,fx_ec,fy_ec,dissipation)
    apply_DG_deriv!(dQ,fx,fy,fxf,fyf,fx_ec,fy_ec,dissipation,rd,cache)

    return nothing
end

# function rhs_scalar_test!(du,fx,fy,fxf,fyf,cache,rd)     
#     @unpack invMQTr,invMQTs = cache
#     @unpack md = cache
#     @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
#     @unpack Vq,Pq,Vf,wf,LIFT = rd   
#     du .= -(rxJ.*(invMQTr*fx) .+ sxJ.*(invMQTs*fx) .+ 
#             ryJ.*(invMQTr*fy) .+ syJ.*(invMQTs*fy) .+
#             LIFT*(@. nxJ*fxf + nyJ*fyf))./J
# end
# function rhs_scalar_loop!(du,fx,fy,fxf,fyf,cache,rd)     
#     @unpack invMQTr,invMQTs = cache
#     @unpack md = cache
#     @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
#     @unpack Vq,Pq,Vf,wf,LIFT = rd   
#     for e = 1:md.K
#         rxJe = @view rxJ[:,e]
#         sxJe = @view sxJ[:,e]
#         ryJe = @view ryJ[:,e]
#         syJe = @view syJ[:,e]
#         Je = @view J[:,e]
#         nxJe = @view nxJ[:,e]
#         nyJe = @view nyJ[:,e]        
#         fxe = @view fx[:,e]
#         fye = @view fy[:,e]        
#         fxfe = @view fxf[:,e]                
#         fyfe = @view fyf[:,e]                        
#         due = @view du[:,e]
#         due .= -(rxJe.*(invMQTr*fxe) .+ sxJe.*(invMQTs*fxe) .+ 
#                  ryJe.*(invMQTr*fye) .+ syJe.*(invMQTs*fye) .+
#                  LIFT*(@. nxJe*fxfe + nyJe*fyfe))./Je
#     end
# end
# function time_rhs_scalar(cache,rd)
#     du,fx,fy = ntuple(x->similar(md.x),3)
#     fxf,fyf = ntuple(x->similar(md.xf),2)
#     @btime rhs_scalar_test!($du,$fx,$fy,$fxf,$fyf,$cache,$rd)
#     @btime rhs_scalar_loop!($du,$fx,$fy,$fxf,$fyf,$cache,$rd)    
# end
################## interface stuff #################

Trixi.ndims(mesh::UnstructuredMesh) = length(mesh.VXYZ)

function Trixi.allocate_coefficients(mesh::UnstructuredMesh, 
                    equations, rd::RefElemData, cache)
    NVARS = nvariables(equations) # TODO replace with static type info?
    return VectorOfArray(Vector([similar(cache.md.x) for i = 1:NVARS])) # TODO replace w/StaticArrays
end

function Trixi.compute_coefficients!(u::VectorOfArray, initial_condition, t, 
                                     mesh::UnstructuredMesh, equations, rd::RefElemData, cache) 
    for i = 1:length(cache.md.x) # loop over nodes
        u0_i = initial_condition(getindex.(cache.md.xyz,i),t,equations) # interpolate
        setindex!.(u.u,u0_i,i)
    end
    # u .= initial_condition(cache.md.xyz,t,equations) # interpolate
    # u .= (x->Pq*x).(initial_condition(cache.md.xyzq,t,equations)) # TODO - projection-based 
end

Trixi.wrap_array(u_ode::VectorOfArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::VectorOfArray, mesh::UnstructuredMesh, equations, solver, cache) = u_ode
Trixi.ndofs(mesh::UnstructuredMesh, rd::RefElemData, cache) = length(rd.r)*cache.md.K

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(UnstructuredMesh((VX,VY),EToV), 
                                    EulerProject2D(f,fEC,v_u,u_v,S), 
                                    initial_condition, rd)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, FinalTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
# analysis_callback = AnalysisCallback(semi, interval=100)

# # The SaveSolutionCallback allows to save the solution to a file in regular intervals
# save_solution = SaveSolutionCallback(interval=100,
#                                      solution_variables=cons2prim)

# # The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
# stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
# callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback)


###############################################################################
# run the simulation
CN = (N+1)*(N+2)/2
dt0 = CFL * sqrt(minimum(md.J)) / CN

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

Q = sol.u[end]
zz = rd.Vp*Q[1]
scatter(rd.Vp*md.x,rd.Vp*md.y,zz,zcolor=zz,leg=false,msw=0,cam=(0,90))

mesh = UnstructuredMesh((VX,VY),EToV)
# eqns = CompressibleEulerEquations2D(1.4)
eqns = EulerProject2D(f,fEC,v_u,u_v,S)
cache = Trixi.create_cache(mesh, eqns, rd, Float64, Float64)
u = Trixi.allocate_coefficients(mesh,eqns,rd,cache)
du = similar(u)
Trixi.compute_coefficients!(u,initial_condition,0.0,mesh,eqns,rd,cache)
Trixi.rhs!(du,u,0.0,mesh,eqns,nothing,nothing,nothing,rd,cache);