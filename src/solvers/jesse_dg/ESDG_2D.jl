using OrdinaryDiffEq
using Trixi
using Plots

using UnPack
using StructArrays
using StaticArrays
using RecursiveArrayTools
using ArrayInterface
using LazyArrays
using LinearAlgebra

using Octavian
using LoopVectorization
using CheapThreads

using StartUpDG

include("/Users/jessechan/.julia/dev/Trixi/src/solvers/jesse_dg/ESDG_utils.jl") # some setup utils

###############################################################################
# semidiscretization 

N = 2
K1D = 16
CFL = .25
FinalTime = 1.0

VX,VY,EToV = uniform_mesh(Tri(),K1D)
rd = RefElemData(Tri(),N)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

struct UnstructuredMesh{NDIMS,Tv,Ti}
    VXYZ::NTuple{NDIMS,Tv}
    EToV::Matrix{Ti}
end
function Base.show(io::IO, md::UnstructuredMesh{NDIMS}) where {NDIMS}
    @nospecialize md
    println("Unstructured mesh in $NDIMS dimensions.")
end

# accumulate Q.*F into rhs
function hadsum_ATr!(rhs,ATr,F::Fxn,u; skip_index=(i,j)->false) where {Fxn}
    rows,cols = axes(ATr)
    for i in cols
        ui = u[i]
        val_i = rhs[i]
        for j in rows
            if !skip_index(i,j)
                val_i .+= ATr[j,i].*F(ui,u[j]) # breaks for tuples, OK for StaticArrays
            end
        end
        rhs[i] = val_i # why not .= here?
    end
end

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
staticzip(::Type{SVector{M}},x::NTuple{M,Array{T,N}}) where {M,T,N} = SVector{M}.(zip(x...))
# xyz = StructArray(staticzip(SVector{2},md.xyz))
# U = (xyz->initial_condition(xyz,0,eqn)).(xyz) # hack to retain tuple elements in U

project_and_store!(y,x) = mul!(y,Ph,x) # can't seem to use matmul! here b/c this is inside a threaded for loop

## ====== workaround for StructArrays/DiffEq.jl from https://github.com/SciML/OrdinaryDiffEq.jl/issues/1386
function RecursiveArrayTools.recursivecopy(a::AbstractArray{T,N}) where {T<:AbstractArray,N}
    if ArrayInterface.ismutable(a)
      b = similar(a)
      map!(recursivecopy,b,a)
    else
      ArrayInterface.restructure(a,map(recursivecopy,a))
    end
end
ArrayInterface.ismutable(x::StructArray) = true
## ====== end workaround ===========

Base.real(rd::RefElemData) = Float64 # is this for DiffEq.jl?

function Trixi.create_cache(mesh::UnstructuredMesh, equations, rd::RefElemData, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # for flux differencing on general elements
    Qrhskew,Qshskew,VhP,Ph = build_hSBP_ops(rd)
    QrhskewTr = Matrix(Qrhskew')
    QshskewTr = Matrix(Qshskew')

    # tmp variables for entropy projection
    Uf = StructArray(staticzip(SVector{4},ntuple(_->similar(md.xf),4)))
    Uq = StructArray(staticzip(SVector{4},ntuple(_->similar(md.xq),4)))
    VUq = similar(Uq)
    VUh = StructArray(staticzip(SVector{4},ntuple(_->similar([md.xq;md.xf]),4)))
    Uh = similar(VUh)

    cache = (;md,
            QrhskewTr,QshskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,Uf)

    return cache
end

@inline function tmap!(f,out,x)
    Trixi.@threaded for i = 1:length(x)
        @inbounds out[i] = f(x[i])
    end
end

## workaround for matmul! with threads https://discourse.julialang.org/t/odd-benchmarktools-timings-using-threads-and-octavian/59838/5
@inline function bmap!(f,out,x)
    @batch for i = 1:length(x)
        @inbounds out[i] = f(x[i])
    end
end


function compute_entropy_projection!(Q,rd::RefElemData,cache,eqn) # why 91.46ms ???
    @unpack Vq = rd    
    @unpack VhP,Ph = cache
    @unpack Uq, VUq, VUh, Uh, Uf = cache

    # entropy projection - should be zero alloc
    StructArrays.foreachfield((uout,u)->matmul!(uout,Vq,u),Uq,Q) # 5μs
    # VUq .= (u->cons2entropy(u,eqn)).(Uq) # 482.656 μs
    bmap!(u->cons2entropy(u,eqn),VUq,Uq) # 77.5μs
    StructArrays.foreachfield((uout,u)->matmul!(uout,VhP,u),VUh,VUq) # 7.902 μs
    # # Uh .= (v->entropy2cons(v,eqn)).(VUh) # 2.148 ms
    bmap!(v->entropy2cons(v,eqn),Uh,VUh) # 327.204 μs
    Nh,Nq = size(VhP)
    Uf .= view(Uh,Nq+1:Nh,:) # 104 μs

    return Uh,Uf
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

    @inline F(orientation) = (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations2D(1.4))

    Uh, Uf = compute_entropy_projection!(Q,rd,cache,equations)
    
    rhse = MVector{4}.(Uh[:,1]) # preallocate storage to be reused     
    for e = 1:md.K
        fill!(rhse,MVector{4}(zeros(4))) # reset contributions
        Ue = view(Uh,:,e)
        QxTr = LazyArray(@~ 2 .*(rxJ[1,e].*QrhskewTr .+ sxJ[1,e].*QshskewTr)) 
        QyTr = LazyArray(@~ 2 .*(ryJ[1,e].*QrhskewTr .+ syJ[1,e].*QshskewTr))

        hadsum_ATr!(rhse, QxTr, F(1), Ue)#; skip_index=(i,j)->i>Nq && j>Nq)
        hadsum_ATr!(rhse, QyTr, F(2), Ue)#; skip_index=(i,j)->i>Nq && j>Nq) 
        
        # add in face contributions
        for (i,vol_id) = enumerate(Nq+1:Nh)
            Fx = F(1)(Uf[mapP[i,e]],Uf[i,e])
            Fy = F(2)(Uf[mapP[i,e]],Uf[i,e])
            dissipation = .25*(Uf[mapP[i,e]] .- Uf[i,e])
            val = @. (Fx * nxJ[i,e] + Fy * nyJ[i,e] - dissipation*sJ[i,e]) * wf[i]
            rhse[vol_id] = rhse[vol_id] .+ val
        end

        # project down and store
        StructArrays.foreachfield(project_and_store!,view(dQ,:,e),rhse)
    end

    return nothing
end

################## interface stuff #################

Trixi.ndims(mesh::UnstructuredMesh) = length(mesh.VXYZ)

function Trixi.allocate_coefficients(mesh::UnstructuredMesh, 
                    equations, rd::RefElemData, cache)
    @unpack md = cache
    NVARS = nvariables(equations) # TODO: replace with static type info?
    return StructArray([MVector{4}(zeros(4)) for i in axes(md.x,1), j in axes(md.x,2)])
end

function Trixi.compute_coefficients!(u::StructArray, initial_condition, t, 
                                     mesh::UnstructuredMesh, equations, rd::RefElemData, cache) 
    for i = 1:length(cache.md.x) # loop over nodes
        xyz_i = getindex.(cache.md.xyz,i)
        u[i] = initial_condition(xyz_i,t,equations) # interpolate
    end
    # u .= initial_condition(cache.md.xyz,t,equations) # interpolate
    # u .= (x->Pq*x).(initial_condition(cache.md.xyzq,t,equations)) # TODO - projection
end

Trixi.wrap_array(u_ode::StructArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::StructArray, mesh::UnstructuredMesh, equations, solver, cache) = u_ode
Trixi.ndofs(mesh::UnstructuredMesh, rd::RefElemData, cache) = length(rd.r)*cache.md.K

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
# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt = dt0, save_everystep=false, callback=callbacks);
# sol = solve(ode, SSPRK43(), dt=.01*dt0, save_everystep=false, callback=callbacks);
sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

# mesh = UnstructuredMesh((VX,VY),EToV)
# eqns = CompressibleEulerEquations2D(1.4)
# cache = Trixi.create_cache(mesh, eqns, rd, Float64, Float64)
# Q = Trixi.allocate_coefficients(mesh,eqns,rd,cache)
# dQ = similar(Q)
# Trixi.compute_coefficients!(Q,initial_condition,0.0,mesh,eqns,rd,cache)
# Trixi.rhs!(dQ,Q,0.0,mesh,eqns,nothing,nothing,nothing,rd,cache);

# Q = sol.u[end]
# zz = rd.Vp*StructArrays.component(dQ,1)
# scatter(rd.Vp*md.x,rd.Vp*md.y,zz,zcolor=zz,leg=false,msw=0,ms=2,cam=(0,90))
