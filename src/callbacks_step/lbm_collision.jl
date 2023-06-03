# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    LBMCollisionCallback()

Apply the Lattice-Boltzmann method (LBM) collision operator before each time step.
See [`LatticeBoltzmannEquations2D`](@ref) for further details.
"""
function LBMCollisionCallback()
    DiscreteCallback(lbm_collision_callback, lbm_collision_callback,
                     save_positions = (false, false),
                     initialize = initialize!)
end

# Always execute collision step after a time step, but not after the last step
lbm_collision_callback(u, t, integrator) = !isfinished(integrator)

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:typeof(lbm_collision_callback)})
    @nospecialize cb # reduce precompilation time

    print(io, "LBMCollisionCallback()")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:typeof(lbm_collision_callback)})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        summary_box(io, "LBMCollisionCallback")
    end
end

# Execute collision step once in the very beginning
function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition,
                                        Affect! <: typeof(lbm_collision_callback)}
    cb.affect!(integrator)
end

# This method is called as callback after the StepsizeCallback during the time integration.
@inline function lbm_collision_callback(integrator)
    dt = get_proposed_dt(integrator)
    semi = integrator.p
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    @unpack collision_op = equations

    u_ode = integrator.u
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    @trixi_timeit timer() "LBM collision" apply_collision!(u, dt, collision_op, mesh,
                                                           equations, solver, cache)

    return nothing
end

include("lbm_collision_dg2d.jl")
include("lbm_collision_dg3d.jl")
end # @muladd
