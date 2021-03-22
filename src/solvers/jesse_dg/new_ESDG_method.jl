using NodesAndModes
using StartUpDG
using Plots
using LinearAlgebra
using ForwardDiff
using StaticArrays
using EntropyStableEuler
using Formatting
using RecursiveArrayTools

using OrdinaryDiffEq
using Trixi
using UnPack

###############################################################################
# semidiscretization of the linear advection equation

N = 5
K1D = 4
CFL = .05
FinalTime = 1.0

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N; quad_rule_vol = gauss_quad(0,0,2*N))
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

struct JesseMesh
    VX
    EToV
end

function u0(x)
    # sine wave
    ρ = @. 1 + .5*sin(2*pi*x)
    u = @. .2 + 0*x
    p = @. 20. + 0*x
#    Q .= VectorOfArray(prim_to_cons(EntropyStableEuler.Euler{1}(),(ρ,u,p)))
    return VectorOfArray(Vector(prim_to_cons(EntropyStableEuler.Euler{1}(),(ρ,u,p)))) # need Vector type for ODE.jl

end
Base.real(rd::RefElemData) = Float64

function Trixi.create_cache(mesh::JesseMesh, equations, rd::RefElemData, RealT, uEltype)
    @unpack VX,EToV = mesh
    md = MeshData(VX,EToV,rd)
    md = make_periodic(md,rd)

    @unpack M,Dr,Vf,Vq,Pq = rd
    Qr = M*Dr
    invMQTr = Matrix(-M\Qr')

    cache = (;md,invMQTr)
    return cache
end

# Euler functions
function f(U)
    ρ,ρu,E = U
    u = ρu./ρ
    p = pfun(EntropyStableEuler.Euler{1}(),U)
    fm = @. ρu*u + p
    fE = @. (E + p)*u
    #return SVector{3}(ρu,fm,fE)
    return [ρu,fm,fE]
end
fEC(uL,uR) = fS(EntropyStableEuler.Euler{1}(),uL,uR)
v_u(u) = v_ufun(EntropyStableEuler.Euler{1}(),u)
u_v(v) = u_vfun(EntropyStableEuler.Euler{1}(),v)
S(u) = Sfun(EntropyStableEuler.Euler{1}(),u)

struct EulerProjection1D
    f
    fEC
    v_u
    u_v
    S
end

function Trixi.rhs!(dU, Q::VectorOfArray, t,
                    mesh::JesseMesh, equations::EulerProjection1D,
                    initial_condition, boundary_conditions, source_terms,
                    rd::RefElemData, cache)

    @unpack md, invMQTr = cache
    @unpack f,fEC,v_u,u_v,S = equations

    @unpack rxJ,J,nxJ,mapP = md
    @unpack M,Vq,Pq,Dr,LIFT,Vf = rd

    U = Q.u # unpack data from VecOfArray

    ṽ = (x->Pq*x).(v_u((x->Vq*x).(U)))
    fN = (x->Pq*x).(f(u_v((x->Vq*x).(ṽ))))
    ff = (x->Vf*x).(fN)
    favg = (u->.5*(u[mapP]+u)).(ff)

    # correction
    uf = u_v((x->Vf*x).(ṽ))
    uP = (u->u[mapP]).(uf)
    fec = fEC(uP,uf)
    Δf = @. fec - favg
    Δv = v_u(uP) .- v_u(uf) # can also compute from ṽ

    # correction
    sign_Δv = map(x->sign.(x),Δv)
    Δf_dot_signΔv = sum(map((x,y)->x.*y.*nxJ,sign_Δv,Δf))
    ctmp = @. max(0,-Δf_dot_signΔv)
    c = map(x->ctmp.*x,sign_Δv) # central dissipation
    # c = map(x->nxJ .* x, -Δf) # EC

    eqn = EntropyStableEuler.Euler{1}()
    λf = sqrt.(pfun(eqn,uf)*eqn.γ ./ uf[1])
    λ = @. .5*max(λf,λf[mapP])
    LFu = map(x->λ .* x,uP .- uf)

    rhsJ(f,flux,c,LFu) = rxJ.*(invMQTr*f) + LIFT*(nxJ.*flux .- c .- LFu)
    dU .= VectorOfArray((x->-x./J).(rhsJ.(fN,favg,c,LFu)))
end

################## interface stuff #################

Trixi.ndims(mesh::JesseMesh) = 1
Trixi.ndims(equations::EulerProjection1D) = 1

function Trixi.allocate_coefficients(mesh::JesseMesh, equations, rd::RefElemData, cache) 
    md = cache.md
    #Q = VectorOfArray(SVector{3}(similar(md.x),similar(md.x),similar(md.x)))
    Q = VectorOfArray([similar(md.x),similar(md.x),similar(md.x)]) # need non-SVector to work w/ODE solver
    return Q
end

function Trixi.compute_coefficients!(Q, u0, t, mesh::JesseMesh, equations, rd::RefElemData, cache) 
    md = cache.md
    @unpack x = md
    Q .= u0(x)
end

Trixi.wrap_array(u_ode::VectorOfArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::VectorOfArray, mesh::JesseMesh, equations, solver, cache) = u_ode
Trixi.ndofs(mesh::JesseMesh, rd::RefElemData, cache) = length(rd.r)*cache.md.K

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(JesseMesh(VX,EToV), EulerProjection1D(f,fEC,v_u,u_v,S), u0, rd)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

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

h = md.J[1]
CN = (N+1)^2/2 

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = CFL*h/CN, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
