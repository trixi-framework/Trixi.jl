using OrdinaryDiffEq
using Trixi
using Plots

using UnPack
using StructArrays
using StructArrays: components
using StaticArrays
using RecursiveArrayTools
using ArrayInterface
using LazyArrays
using LinearAlgebra
using BenchmarkTools

using Octavian
using LoopVectorization
using CheapThreads

using StartUpDG

include("ESDG_utils.jl") # some setup utils
include("temporary_structarrays_support.jl") # until https://github.com/JuliaArrays/StructArrays.jl/pull/186 is merged

###############################################################################
# semidiscretization

N = 3
K1D = 8
CFL = .1
FinalTime = .250

VX,VY,EToV = uniform_mesh(Tri(),K1D)
rd = RefElemData(Tri(),N)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

function initial_condition(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    ρ = 1 + .5*exp(-25*(x^2+y^2))
    # ρ = 1 + .5*sin(pi*x)*sin(pi*y)
    u = 1.0
    v = .5
    p = 2.
    return prim2cons((ρ,u,v,p),equations)
end

eqn = CompressibleEulerEquations2D(1.4)
F(orientation) = (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations2D(1.4))
Qrhskew,Qshskew,VhP,Ph = build_hSBP_ops(rd)
QrhskewTr = Matrix(Qrhskew')
QshskewTr = Matrix(Qshskew')

# StructArray initialization - problem in entropy2cons

function Trixi.create_cache(mesh::UnstructuredMesh, equations, rd::RefElemData, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # for flux differencing on general elements
    Qrhskew,Qshskew,VhP,Ph = build_hSBP_ops(rd)
    QrhskewTr = Matrix(Qrhskew')
    QshskewTr = Matrix(Qshskew')

    # tmp variables for entropy projection
    # Uq = StructArray(SVector{4}.(ntuple(_->similar(md.xq),4)))
    Uq = StructArray{SVector{4,Float64}}(ntuple(_->similar(md.xq),4))
    VUq = similar(Uq)
    VUh = StructArray{SVector{4,Float64}}(ntuple(_->similar([md.xq;md.xf]),4))
    Uh = similar(VUh)

    cache = (;md,
            QrhskewTr,QshskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh)

    return cache
end

let Ph=Ph
    global project_and_store!(y,x) = mul!(y,Ph,x)
end

@inline function compute_entropy_projection!(Q,rd::RefElemData,cache,eqn)
    @unpack Vq = rd
    @unpack VhP,Ph = cache
    @unpack Uq, VUq, VUh, Uh = cache

    # # entropy projection - should be zero alloc
    # StructArrays.foreachfield((uout,u)->mul!(uout,Vq,u),Uq,Q)
    # tmap!(u->cons2entropy(u,eqn),VUq,Uq)
    # StructArrays.foreachfield((uout,u)->mul!(uout,VhP,u),VUh,VUq)
    # tmap!(v->entropy2cons(v,eqn),Uh,VUh)

    # map((uout,u)->matmul!(uout,Vq,u),components(Uq),components(Q))
    StructArrays.foreachfield((uout,u)->matmul!(uout,Vq,u),Uq,Q)
    bmap!(u->cons2entropy(u,eqn),VUq,Uq) # 77.5μs
    StructArrays.foreachfield((uout,u)->matmul!(uout,VhP,u),VUh,VUq)
    # map((uout,u)->matmul!(uout,VhP,u),components(VUh),components(VUq))
    bmap!(v->entropy2cons(v,eqn),Uh,VUh) # 327.204 μs

    Nh,Nq = size(VhP)
    Uf = view(Uh,Nq+1:Nh,:) # 24.3 μs

    return Uh,Uf
end

@inline function max_abs_speed_normal(UL, UR, normal, equations::CompressibleEulerEquations2D)
    # Calculate primitive variables and speed of sound
    ρu_n_L = UL[2]*normal[1] + UL[3]*normal[2]
    ρu_n_R = UR[2]*normal[1] + UR[3]*normal[2]
    uL = (UL[1],ρu_n_L,UL[4])
    uR = (UR[1],ρu_n_R,UR[4])
    return Trixi.max_abs_speed_naive(uL,uR,0,CompressibleEulerEquations1D(equations.gamma))
end

function Trixi.rhs!(dQ, Q::StructArray, t,
                    mesh::UnstructuredMesh, equations::CompressibleEulerEquations2D,
                    initial_condition, boundary_conditions, source_terms,
                    rd::RefElemData, cache)

    @unpack md = cache
    @unpack QrhskewTr,QshskewTr,VhP,Ph = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack Vq,wf = rd

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i>Nq && j > Nq
    end

    @inline F(orientation) = let equations = equations
        # (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,equations)
        (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations2D(1.4))
    end

    Trixi.@timeit_debug Trixi.timer() "compute_entropy_projection!" Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs

    zero_vec = SVector(ntuple(_ -> 0.0, Val(nvariables(equations))))
    # rhse = similar(Uh[:,1])
    # for e = 1:md.K
    Trixi.@timeit_debug Trixi.timer() "rhse_threads" rhse_threads = StructVector{SVector{nvariables(equations), Float64}, NTuple{nvariables(equations), Vector{Float64}}, Int}[SVector{4}.(Uh[:,1]) for _ in 1:Threads.nthreads()]
    @batch for e = 1:md.K
        rhse = rhse_threads[Threads.threadid()]

        fill!(rhse,zero_vec) # 40ns, (1 allocation: 48 bytes)
        Ue = view(Uh,:,e)    # 8ns (0 allocations: 0 bytes) after @inline in Base.view(StructArray)
        Trixi.@timeit_debug Trixi.timer() "QxTr, QyTr" begin
            QxTr = LazyArray(@~ @. 2 *(rxJ[1,e]*QrhskewTr + sxJ[1,e]*QshskewTr))
            QyTr = LazyArray(@~ @. 2 *(ryJ[1,e]*QrhskewTr + syJ[1,e]*QshskewTr))
        end

        Trixi.@timeit_debug Trixi.timer() "hadsum_ATr!" begin
            hadsum_ATr!(rhse, QxTr, F(1), Ue, skip_index)
            hadsum_ATr!(rhse, QyTr, F(2), Ue, skip_index)
        end

        Trixi.@timeit_debug Trixi.timer() "loop" for (i,vol_id) = enumerate(Nq+1:Nh)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]
            Fx = F(1)(UP,UM)
            Fy = F(2)(UP,UM)
            λ = max_abs_speed_normal(UP, UM, SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e], equations)
            val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] - .5*λ*(UP - UM)*sJ[i,e]) * wf[i]
            rhse[vol_id] += val
        end

        # project down and store
        Trixi.@timeit_debug Trixi.timer() "project_and_store!" begin
            @. rhse = -rhse / J[1,e]
            StructArrays.foreachfield(project_and_store!, view(dQ,:,e), rhse)
        end
    end

    return nothing
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(UnstructuredMesh((VX,VY),EToV), CompressibleEulerEquations2D(1.4),
                                    initial_condition, rd)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, FinalTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)


###############################################################################
# run the simulation
CN = (N+1)*(N+2)/2
dt0 = CFL * sqrt(minimum(md.J)) / CN

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
# sol = solve(ode, SSPRK43(), dt=.01*dt0, save_everystep=false, callback=callbacks);
sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

# mesh = UnstructuredMesh((VX,VY),EToV)
# eqns = CompressibleEulerEquations2D(1.4)
# cache = Trixi.create_cache(mesh, eqns, rd, Float64, Float64)
# Q = Trixi.allocate_coefficients(mesh,eqns,rd,cache)
# dQ = zero(Q)
# Trixi.compute_coefficients!(Q,initial_condition,0.0,mesh,eqns,rd,cache)

# # dump internals
# @unpack md = cache
# @unpack QrhskewTr,QshskewTr,VhP,Ph = cache
# @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
# @unpack Vq,wf = rd
# Nh,Nq = size(VhP)
# @inline F(orientation) = let equations = equations
#     # (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,equations)
#     (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations2D(1.4))
# end
# Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs
# rhse = similar(Uh[:,1])
# e = 1
# fill!(rhse,zero_vec) # 40ns, (1 allocation: 48 bytes)
# Ue = view(Uh,:,e)    # 8ns (0 allocations: 0 bytes) after @inline in Base.view(StructArray)
# QxTr = LazyArray(@~ 2 .*(rxJ[1,e].*QrhskewTr .+ sxJ[1,e].*QshskewTr))
# QyTr = LazyArray(@~ 2 .*(ryJ[1,e].*QrhskewTr .+ syJ[1,e].*QshskewTr))
# # Trixi.rhs!(dQ,Q,0.0,mesh,eqns,nothing,nothing,nothing,rd,cache);

Q = sol.u[end]
zz = rd.Vp*StructArrays.component(Q,1)
scatter(rd.Vp*md.x,rd.Vp*md.y,zz,zcolor=zz,leg=false,msw=0,ms=2,cam=(0,90),ratio=1)
