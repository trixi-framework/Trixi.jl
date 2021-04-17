using OrdinaryDiffEq
using Trixi
using Plots

using UnPack
using StructArrays
using ArraysOfArrays 
using RecursiveArrayTools
using LazyArrays
using LinearAlgebra

using Octavian
using LoopVectorization

using StartUpDG

# using EntropyStableEuler

include("./ESDG_utils.jl") # some setup utils

###############################################################################
# semidiscretization 

N = 2
K1D = 32
CFL = .1
FinalTime = .50

VX,VY,EToV = uniform_mesh(Tri(),K1D)
rd = RefElemData(Tri(),N)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

struct UnstructuredMesh{NDIMS,Tv,Ti}
    VXYZ::NTuple{NDIMS,Tv}
    EToV::Matrix{Ti}
end

# accumulate Q.*F into rhs
function hadsum_ATr!(rhs,ATr,F::Fxn,u; skip_index=(i,j)->false) where {Fxn}
    val_i = zero(eltype(rhs)) # preallocate array
    rows,cols = axes(ATr)
    for i in cols
        ui = getindex(u,i)
        val_i = getindex(rhs,i)
        for j in rows
            if !skip_index(i,j)
                uj = getindex(u,j)
                val_i += ATr[j,i]*F(ui,uj)
            end
        end
        setindex!(rhs,val_i,i)
    end
end

function initial_condition(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    ρ = 1. + .5*exp(-25*(x^2+y^2))
    u = 1.0
    v = .5    
    p = 2.
    return prim2cons((ρ,u,v,p),equations)
end

xyz = StructArray(md.xyz)
eqn = CompressibleEulerEquations2D(1.4)

# ArraysOfArrays initialization



# structarray initialization - problem in entropy2cons
U = (xyz->tuple(initial_condition(xyz,0,eqn)...)).(xyz) # hack to retain tuple elements in U

tmp_vol_quad = StructArray(ntuple(x->similar(md.xq),Trixi.nvariables(eqn))) # storage at volume quadrature points
tmp_face_quad = StructArray(ntuple(x->similar(md.xf),Trixi.nvariables(eqn))) # storage at face quadrature points
VqPq = rd.Vq*rd.Pq # project + evaluate at quad points

# entropy projection - should be zero alloc
StructArrays.foreachfield((u,uout)->matmul!(uout,rd.Vq,u),U,tmp_vol_quad)
tmp_vol_quad .= (u->tuple(cons2entropy(u,eqn)...)).(tmp_vol_quad)
StructArrays.foreachfield((u,uout)->matmul!(uout,VqPq,u),U,tmp_vol_quad)
tmp_vol_quad .= (v->tuple(entropy2cons(v,eqn)...)).(tmp_vol_quad) # issue - w * (γ-1) in entropy2cons fails for w::Tuple


Base.real(rd::RefElemData) = Float64 # is this for DiffEq.jl?

function Trixi.create_cache(mesh::UnstructuredMesh, equations, rd::RefElemData, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # for flux differencing on general elements
    Qrhskew,Qshskew,VhP,Ph = build_hSBP_ops(rd)
    QrhskewTr = Matrix(Qrhskew')
    QshskewTr = Matrix(Qshskew')

    cache = (;md,QrhskewTr,QshskewTr,VhP,Ph)

    return cache
end

function compute_entropy_projection(Q,rd::RefElemData,cache,equations)
    @unpack Vq = rd    
    @unpack QrhskewTr,QshskewTr,VhP,Ph = cache

    # entropy projection
    VU = (x->VhP*x).(v_ufun(EntropyStableEuler.Euler{2}(),(x->Vq*x).(Q)))
    Uh = u_vfun(EntropyStableEuler.Euler{2}(),VU)    
end

function Trixi.rhs!(du, u::VectorOfArray, t,
                    mesh::UnstructuredMesh, equations::CompressibleEulerEquations2D,
                    initial_condition, boundary_conditions, source_terms,
                    rd::RefElemData, cache)

    @unpack md = cache
    @unpack QrhskewTr,QshskewTr,VhP,Ph = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack Vq,wf = rd

    # unpack VecOfArray storage for bcast    
    Q = u.u 
    dQ = du.u

    Nh,Nq = size(VhP)

    @inline flux_x(uL,uR) = flux_chandrashekar(uL,uR,1,equations)
    @inline flux_y(uL,uR) = flux_chandrashekar(uL,uR,2,equations)

    Uh = compute_entropy_projection(Q,rd,cache,equations)

    # compute face values
    Uf = (x->x[Nq+1:Nh,:]).(Uh)
    
    rhse = similar.(getindex.(Uh,:,1))
    for e = 1:md.K
        fill!.(rhse,zero(eltype(first(rhse)))) # reset contributions
        Ue = view.(Uh,:,e)
        Qx = LazyArray(@~ 2 .*(rxJ[1,e].*QrhskewTr .+ sxJ[1,e].*QshskewTr)) 
        Qy = LazyArray(@~ 2 .*(ryJ[1,e].*QrhskewTr .+ syJ[1,e].*QshskewTr))

        hadsum_ATr!(rhse, Qx, flux_x, Ue; skip_index=(i,j)->i>Nq && j>Nq)
        hadsum_ATr!(rhse, Qy, flux_y, Ue; skip_index=(i,j)->i>Nq && j>Nq) 
        
        # add in face contributions
        for (i,vol_id) = enumerate(Nq+1:Nh)
            Fx = flux_x(getindex.(Uf,mapP[i,e]),getindex.(Uf,i,e))
            Fy = flux_y(getindex.(Uf,mapP[i,e]),getindex.(Uf,i,e))
            dissipation = .25*(getindex.(Uf,mapP[i,e]) .- getindex.(Uf,i,e))
            val = @. (Fx * nxJ[i,e] + Fy * nyJ[i,e] - dissipation*sJ[i,e]) * wf[i]
            setindex!.(rhse, getindex.(rhse,vol_id) .+ val, vol_id)
        end

        # project down and store
        setindex!.(dQ,(x->-(Ph*x)./J[1,e]).(rhse),:,e)
    end

    return nothing
end

################## interface stuff #################

Trixi.ndims(mesh::UnstructuredMesh) = length(mesh.VXYZ)

function Trixi.allocate_coefficients(mesh::UnstructuredMesh, 
                    equations, rd::RefElemData, cache)
    NVARS = nvariables(equations) # TODO replace with static type info 
    return VectorOfArray(Vector([similar(cache.md.x) for i = 1:NVARS]))
end

function Trixi.compute_coefficients!(u::VectorOfArray, initial_condition, t, 
                                     mesh::UnstructuredMesh, equations, rd::RefElemData, cache) 
    for i = 1:length(cache.md.x) # loop over nodes
        u0_i = initial_condition(getindex.(cache.md.xyz,i),t,equations) # interpolate
        setindex!.(u.u,u0_i,i)
    end
    # u .= initial_condition(cache.md.xyz,t,equations) # interpolate
    # u .= (x->Pq*x).(initial_condition(cache.md.xyzq,t,equations)) # TODO - projection
end

Trixi.wrap_array(u_ode::VectorOfArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::VectorOfArray, mesh::UnstructuredMesh, equations, solver, cache) = u_ode
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

Q = sol.u[end]
zz = rd.Vp*Q[1]
scatter(rd.Vp*md.x,rd.Vp*md.y,zz,zcolor=zz,leg=false,msw=0,cam=(0,90))

# # LaxFriedrichs(flux_ranocha)
# mesh = UnstructuredMesh((VX,VY),EToV)
# eqns = CompressibleEulerEquations2D(1.4)
# cache = Trixi.create_cache(mesh, eqns, rd, Float64, Float64)
# Q = Trixi.allocate_coefficients(mesh,eqns,rd,cache)
# dQ = similar(Q)
# Trixi.compute_coefficients!(Q,initial_condition,0.0,mesh,eqns,rd,cache)
# Trixi.rhs!(dQ,Q,0.0,mesh,eqns,nothing,nothing,nothing,rd,cache);
