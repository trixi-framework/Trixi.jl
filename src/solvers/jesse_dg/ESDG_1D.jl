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
K1D = 32
CFL = .01
FinalTime = .7

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N)
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

Qrhskew,VhP,Ph = build_hSBP_ops(rd)
project_and_store!(y,x) = let Ph=Ph
    mul!(y,Ph,x) # can't use matmul! b/c its applied to a subarray
end


function initial_condition(xyz,t,equations::CompressibleEulerEquations1D)
    x, = xyz
    # ρ = 1 + .5*exp(-25*(x^2+y^2))
    ρ = 1 + .99*sin(pi*x)
    u = 1.0
    p = 2.
    return prim2cons((ρ,u,p),equations)
end

eqn = CompressibleEulerEquations1D(1.4)
F(orientation) = (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations1D(1.4))

# StructArray initialization - problem in entropy2cons

function Trixi.create_cache(mesh::UnstructuredMesh, equations::CompressibleEulerEquations1D, 
                            rd::RefElemData, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # for flux differencing on general elements
    Qrhskew,VhP,Ph = build_hSBP_ops(rd)
    QrhskewTr = Matrix(Qrhskew')

    # tmp variables for entropy projection
    Uq = StructArray{SVector{3,Float64}}(ntuple(_->similar(md.xq),nvariables(equations)))
    VUq = similar(Uq)
    VUh = StructArray{SVector{3,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvariables(equations)))
    Uh = similar(VUh)

    cache = (;md,
            QrhskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh)

    return cache
end

@inline function compute_entropy_projection!(Q,rd::RefElemData,cache,eqn::CompressibleEulerEquations1D) 
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

function Trixi.rhs!(dQ, Q::StructArray, t,
                    mesh::UnstructuredMesh, equations::CompressibleEulerEquations1D,
                    initial_condition, boundary_conditions, source_terms,
                    rd::RefElemData, cache)

    @unpack md = cache
    @unpack QrhskewTr,VhP,Ph = cache
    @unpack rxJ,J,nxJ,sJ,mapP = md
    @unpack Vq,wf = rd

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i>Nq && j > Nq
    end

    @inline F(orientation) = let equations = equations 
        # (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,equations)
        (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations1D(1.4))
    end

    Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs
    
    zero_vec = SVector{3}(zeros(nvariables(equations)))
    rhse_threads = [SVector{3}.(Uh[:,1]) for _ in 1:Threads.nthreads()]
    # rhse = similar(Uh[:,1])
    # for e = 1:md.K
    @batch for e = 1:md.K 
        rhse = rhse_threads[Threads.threadid()]

        fill!(rhse,zero_vec) # 40ns, (1 allocation: 48 bytes)
        Ue = view(Uh,:,e)    # 8ns (0 allocations: 0 bytes) after @inline in Base.view(StructArray)
        QxTr = LazyArray(@~ @. 2 * rxJ[1,e]*QrhskewTr )

        hadsum_ATr!(rhse, QxTr, F(1), Ue; skip_index=skip_index) # 8.274 μs (15 allocations: 720 bytes). Slower with skip_index?
        
        for (i,vol_id) = enumerate(Nq+1:Nh)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]        
            Fx = F(1)(UP,UM)
            λ = max_abs_speed_naive(UP, UM, SVector{1}(nxJ[i,e]), equations) 
            val = @. (Fx * nxJ[i,e] - .5*λ*(UP - UM)*sJ[i,e]) 
            rhse[vol_id] = rhse[vol_id] + val
        end

        # project down and store
        StructArrays.foreachfield(project_and_store!,view(dQ,:,e),-rhse/J[1,e]) # 2.997 μs
    end

    return nothing
end

###############################################################################
# ODE solvers, callbacks etc.

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(UnstructuredMesh((VX,),EToV), CompressibleEulerEquations1D(1.4), 
                                    initial_condition, rd)

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

# mesh = UnstructuredMesh((VX,),EToV)
# eqns = CompressibleEulerEquations1D(1.4)
# cache = Trixi.create_cache(mesh, eqns, rd, Float64, Float64)
# Q = Trixi.allocate_coefficients(mesh,eqns,rd,cache)
# dQ = zero(Q)
# Trixi.compute_coefficients!(Q,initial_condition,0.0,mesh,eqns,rd,cache)

Q = sol.u[end]
zz = rd.Vp*StructArrays.component(Q,1)
scatter(rd.Vp*md.x,zz,leg=false,msw=0,ms=2,ratio=1)
