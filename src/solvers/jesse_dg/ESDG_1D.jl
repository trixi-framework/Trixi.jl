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
include("flux_differencing.jl")
include("ModalESDG.jl")

###############################################################################
# semidiscretization 

N = 3
K1D = 32
CFL = .05
FinalTime = .7

rd = RefElemData(Line(),N)
VX,EToV = uniform_mesh(Line(),K1D)
md = MeshData(VX,EToV,rd)

eqn = CompressibleEulerEquations1D(1.4)
function LxF_dissipation(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)
    return .5 * max_abs_speed_naive(u_ll, u_rr, orientation, equations) * (u_ll-u_rr)
end
solver = ModalESDG(rd,Trixi.flux_chandrashekar,Trixi.flux_chandrashekar,LxF_dissipation,eqn)

Qrhskew,VhP,Ph = hybridized_SBP_operators(rd)
project_and_store! = let Ph=Ph
    (y,x)->mul!(y,Ph,x) # can't use matmul! b/c its applied to a subarray
end

function initial_condition(xyz,t,equations::CompressibleEulerEquations1D)
    x, = xyz
    # ρ = 1 + .5*exp(-25*(x^2+y^2))
    ρ = 1 + .99*sin(pi*x)
    u = 1.0
    p = 2.
    return prim2cons((ρ,u,p),equations)
end

function Trixi.create_cache(mesh::UnstructuredMesh, equations::CompressibleEulerEquations1D, 
                            solver::ModalESDG, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # make skew symmetric versions of the operators"
    Qrh,VhP,Ph = hybridized_SBP_operators(rd)
    Qrhskew = .5*(Qrh-transpose(Qrh))
    QrhskewTr = typeof(Qrh)(Qrhskew')

    # tmp variables for entropy projection
    nvars = nvariables(equations)
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))
    VUq = similar(Uq)
    VUh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars))
    Uh = similar(VUh)

    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    cache = (;md,
            QrhskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,
            rhse_threads)

    return cache
end

function Trixi.rhs!(dQ, Q::StructArray, t,
                    mesh::UnstructuredMesh, equations::CompressibleEulerEquations1D,
                    initial_condition, boundary_conditions, source_terms,
                    solver::ModalESDG, cache)

    @unpack md = cache
    @unpack QrhskewTr,VhP,Ph = cache
    @unpack rxJ,J,nxJ,sJ,mapP = md
    rd = solver.rd
    @unpack Vq,wf = rd
    @unpack volume_flux, interface_flux, interface_dissipation = solver

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i>Nq && j > Nq
    end

    Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs
        
    @batch for e = 1:md.K 
        rhse = cache.rhse_threads[Threads.threadid()]

        fill!(rhse,zero(eltype(rhse))) # 40ns, (1 allocation: 48 bytes)
        Ue = view(Uh,:,e)    # 8ns (0 allocations: 0 bytes) after @inline in Base.view(StructArray)
        QxTr = LazyArray(@~ @. 2 * rxJ[1,e]*QrhskewTr )

        hadsum_ATr!(rhse, QxTr, volume_flux(1), Ue, skip_index) 
        
        # add in interface flux contributions
        for (i,vol_id) = enumerate(Nq+1:Nh)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]        
            Fx = interface_flux(1)(UP,UM)
            diss = interface_dissipation(SVector{1}(nxJ[i,e]))(UM,UP)
            val = (Fx * nxJ[i,e] + diss*sJ[i,e]) 
            rhse[vol_id] = rhse[vol_id] + val
        end

        # project down and store
        @. rhse = -rhse/J[1,e]
        StructArrays.foreachfield(project_and_store!,view(dQ,:,e),rhse) 
    end

    return nothing
end

###############################################################################
# ODE solvers, callbacks etc.

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(UnstructuredMesh((VX,),EToV), CompressibleEulerEquations1D(1.4), 
                                    initial_condition, solver)

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, FinalTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation
CN = (N+1)*(N+2)/2
dt0 = CFL * 2. / (K1D*CN)

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
# sol = solve(ode, SSPRK43(), dt=.01*dt0, save_everystep=false, callback=callbacks);
sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

Q = sol.u[end]
zz = rd.Vp*StructArrays.component(Q,1)
scatter(rd.Vp*md.x,zz,leg=false,msw=0,ms=2,ratio=1)
