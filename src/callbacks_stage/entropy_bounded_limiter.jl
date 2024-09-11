# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    EntropyBoundedLimiter(; density_threshold)
"""
mutable struct EntropyBoundedLimiter{RealT <: Real}
    density_threshold::RealT
    min_entropy_exp::Vector{RealT}
end

function EntropyBoundedLimiter(; density_threshold)
    EntropyBoundedLimiter(density_threshold, Vector{eltype(density_threshold)}())
end

function (limiter!::EntropyBoundedLimiter)(u_ode,
                                          integrator::Trixi.SimpleIntegratorSSP,
                                          stage)
    limiter!(u_ode, integrator.p)
end

function (limiter!::EntropyBoundedLimiter)(u_ode, semi::AbstractSemidiscretization)
    u = wrap_array(u_ode, semi)
    @trixi_timeit timer() "entropy-bounded limiter" begin
        limiter_entropy_bounded!(u, limiter!.density_threshold, limiter!.min_entropy_exp, 
                           mesh_equations_solver_cache(semi)...)
    end
end

function prepare_callback!(limiter::EntropyBoundedLimiter, integrator)
    semi = integrator.p
    u = wrap_array(integrator.u, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Compute the minimum entropy exponent for each node
    limiter.min_entropy_exp = zeros(eltype(u), nelements(dg, cache))
    prepare_limiter!(limiter, mesh, equations, dg, cache, u)
end

init_callback(limiter!::EntropyBoundedLimiter, semi) = nothing
finalize_callback(limiter!::EntropyBoundedLimiter, semi) = nothing

@inline function entropy_difference(p, entropy_exp, rho, gamma)
    return p - entropy_exp * rho^gamma
end

include("entropy_bounded_limiter_1d.jl")
end # @muladd
