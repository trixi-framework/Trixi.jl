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

include("trixi_interface.jl") # some setup utils
include("ModalESDG.jl")

###############################################################################
# semidiscretization

N = 2
K1D = 16
CFL = .1
FinalTime = .250

VX,VY,EToV = uniform_mesh(Tri(),K1D)
rd = RefElemData(Tri(),N)
md = MeshData(VX,VY,EToV,rd)

eqn = CompressibleEulerEquations2D(1.4)
@inline function max_abs_speed_normal(UL, UR, normal, equations::CompressibleEulerEquations2D)
    # Calculate primitive variables and speed of sound
    ρu_n_L = UL[2]*normal[1] + UL[3]*normal[2]
    ρu_n_R = UR[2]*normal[1] + UR[3]*normal[2]
    uL = (UL[1],ρu_n_L,UL[4])
    uR = (UR[1],ρu_n_R,UR[4])
    return Trixi.max_abs_speed_naive(uL,uR,0,CompressibleEulerEquations1D(equations.gamma))
end
@inline function LxF_dissipation(u_ll, u_rr, normal, equations::CompressibleEulerEquations2D)
    return .5 * max_abs_speed_normal(u_ll, u_rr, normal, equations) * (u_ll-u_rr)
end
solver = ModalESDG(rd,Trixi.flux_chandrashekar,Trixi.flux_chandrashekar,LxF_dissipation,eqn)

function initial_condition(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    ρ = 1 + .5*exp(-25*(x^2+y^2))
    # ρ = 1 + .5*sin(pi*x)*sin(pi*y)
    u = 1.0
    v = .5
    p = 2.
    return prim2cons((ρ,u,v,p),equations)
end

function Trixi.create_cache(mesh::UnstructuredMesh, equations, solver::ModalESDG, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # for flux differencing on general elements
    Qrhskew,Qshskew,VhP,Ph = hybridized_SBP_operators(rd)
    QrhskewTr = Matrix(Qrhskew')
    QshskewTr = Matrix(Qshskew')

    # tmp variables for entropy projection
    nvars = nvariables(equations)
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))
    VUq = similar(Uq)
    VUh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars))
    Uh = similar(VUh)

    # tmp cache for threading
    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    cache = (;md,
            QrhskewTr,QshskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,
            rhse_threads)

    return cache
end

Qrhskew,Qshskew,VhP,Ph = hybridized_SBP_operators(rd)
project_and_store! = let Ph=Ph
    (y,x)->mul!(y,Ph,x)
end



function Trixi.rhs!(dQ, Q::StructArray, t,
                    mesh::UnstructuredMesh, equations::CompressibleEulerEquations2D,
                    initial_condition, boundary_conditions, source_terms,
                    solver::ModalESDG, cache)

    @unpack md = cache
    @unpack QrhskewTr,QshskewTr,VhP,Ph = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    rd = solver.rd
    @unpack Vq,wf = rd
    @unpack volume_flux, interface_flux, interface_dissipation = solver

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i>Nq && j > Nq
    end

    Trixi.@timeit_debug Trixi.timer() "compute_entropy_projection!" begin
        Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs
    end

    @batch for e = 1:md.K
        rhse = cache.rhse_threads[Threads.threadid()]

        fill!(rhse,zero(eltype(rhse)))
        Ue = view(Uh,:,e)   
        QxTr = LazyArray(@~ @. 2 *(rxJ[1,e]*QrhskewTr + sxJ[1,e]*QshskewTr))
        QyTr = LazyArray(@~ @. 2 *(ryJ[1,e]*QrhskewTr + syJ[1,e]*QshskewTr))        

        hadsum_ATr!(rhse, QxTr, volume_flux(1), Ue, skip_index) 
        hadsum_ATr!(rhse, QyTr, volume_flux(2), Ue, skip_index) 

        for (i,vol_id) = enumerate(Nq+1:Nh)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]
            Fx = interface_flux(1)(UP,UM)
            Fy = interface_flux(2)(UP,UM)
            normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
            diss = interface_dissipation(normal)(UM,UP)
            val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
            rhse[vol_id] += val
        end

        # project down and store
        @. rhse = -rhse / J[1,e]
        StructArrays.foreachfield(project_and_store!, view(dQ,:,e), rhse)
    end

    return nothing
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(UnstructuredMesh((VX,VY),EToV), CompressibleEulerEquations2D(1.4),
                                    initial_condition, solver)

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
# sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)
sol = solve(ode,CarpenterKennedy2N54(williamson_condition=false), dt=dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

Q = sol.u[end]
zz = rd.Vp*StructArrays.component(Q,1)
scatter(rd.Vp*md.x,rd.Vp*md.y,zz,zcolor=zz,leg=false,msw=0,ms=2,cam=(0,90),ratio=1)
