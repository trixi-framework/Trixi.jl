# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO: Version with indicator
"""
    EntropyBoundedLimiter{RealT <: Real}
"""
mutable struct EntropyBoundedLimiter{RealT <: Real}
    entropy_decrease_max::RealT # < 0
    # `resize!`able storage for the minimum exponentiated entropy per element.
    min_entropy_exp::Vector{RealT}
end

# This constructor sets only the element type of the `min_entropy_exp` vector
function EntropyBoundedLimiter(; entropy_decrease_max::RealT) where {RealT <: Real}
    @assert entropy_decrease_max<0 "Supplied `entropy_decrease_max` expected to be negative"
    EntropyBoundedLimiter{RealT}(entropy_decrease_max, Vector{RealT}())
end

function (limiter!::EntropyBoundedLimiter)(u_ode,
                                           integrator::Trixi.SimpleIntegratorSSP,
                                           stage)
    limiter!(u_ode, integrator.p)
end

function (limiter!::EntropyBoundedLimiter)(u_ode, semi::AbstractSemidiscretization)
    u = wrap_array(u_ode, semi)
    @trixi_timeit timer() "entropy-bounded limiter" begin
        limiter_entropy_bounded!(u, limiter!.entropy_decrease_max,
                                 limiter!.min_entropy_exp,
                                 mesh_equations_solver_cache(semi)...)
    end
end

# Store previous iterates' minimum exponentiated entropy per element
function prepare_callback!(limiter::EntropyBoundedLimiter, integrator)
    semi = integrator.p
    u = wrap_array(integrator.u, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Compute the minimum exponentiated entropy `exp(s)` for each node
    limiter.min_entropy_exp = zeros(eltype(u), nelements(dg, cache))
    save_min_exp_entropy!(limiter, mesh, equations, dg, cache, u)
end

# This is called after the mesh has changed (AMR)
function Base.resize!(limiter::EntropyBoundedLimiter, nelements)
    resize!(limiter.min_entropy_exp, nelements)
end

function init_callback(limiter::EntropyBoundedLimiter, semi)
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    resize!(limiter.min_entropy_exp, nelements(dg, cache))
end

finalize_callback(limiter::EntropyBoundedLimiter, semi) = nothing

# Entropy increase for the thermodynamic entropy (see `entropy_thermodynamic`) 
# of an ideal gas with constant gamma.
@inline function entropy_increase(p, entropy_exp, rho, gamma)
    return p - entropy_exp * rho^gamma
end

include("entropy_bounded_limiter_1d.jl")
include("entropy_bounded_limiter_2d.jl")
include("entropy_bounded_limiter_3d.jl")
end # @muladd
