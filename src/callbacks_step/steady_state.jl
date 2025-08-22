# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the [`residual_steady_state(du, equations)`](@ref)
falls below the threshold specified by `abstol, reltol`.
"""
mutable struct SteadyStateCallback{RealT <: Real}
    abstol::RealT
    reltol::RealT
end

function SteadyStateCallback(; abstol = 1.0e-8, reltol = 1.0e-6)
    abstol, reltol = promote(abstol, reltol)
    steady_state_callback = SteadyStateCallback(abstol, reltol)

    DiscreteCallback(steady_state_callback, steady_state_callback,
                     save_positions = (false, false))
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    steady_state_callback = cb.affect!
    print(io, "SteadyStateCallback(abstol=", steady_state_callback.abstol, ", ",
          "reltol=", steady_state_callback.reltol, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        steady_state_callback = cb.affect!

        setup = [
            "absolute tolerance" => steady_state_callback.abstol,
            "relative tolerance" => steady_state_callback.reltol
        ]
        summary_box(io, "SteadyStateCallback", setup)
    end
end

# affect!
(::SteadyStateCallback)(integrator) = terminate!(integrator)

# the condition
function (steady_state_callback::SteadyStateCallback)(u_ode, t, integrator)
    semi = integrator.p

    u = wrap_array(u_ode, semi)
    du = wrap_array(get_du(integrator), semi)
    terminate = steady_state_callback(du, u, semi)
    if mpi_isparallel()
        # MPI.jl doesn't seem to have MPI_C_BOOL
        terminate_integer = Int(terminate)
        terminate = !iszero(MPI.Allreduce!(Ref(terminate_integer), +, mpi_comm())[])
    end
    if mpi_isroot() && terminate
        @info "  Steady state tolerance reached" steady_state_callback t
    end
    return terminate
end

function (steady_state_callback::SteadyStateCallback)(du, u,
                                                      semi::AbstractSemidiscretization)
    steady_state_callback(du, u, mesh_equations_solver_cache(semi)...)
end

include("steady_state_dg1d.jl")
include("steady_state_dg2d.jl")
include("steady_state_dg3d.jl")
end # @muladd
