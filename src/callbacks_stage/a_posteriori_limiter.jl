# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    APosterioriLimiter()

Perform antidiffusive stage for a posteriori IDP limiting.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct APosterioriLimiter end

function (limiter!::APosterioriLimiter)(u_ode, integrator::Trixi.SimpleIntegratorSSP,
                                        stage)
    limiter!(u_ode, integrator.p, integrator.t, integrator.dt,
             integrator.p.solver.volume_integral)
end

function (::APosterioriLimiter)(u_ode, semi, t, dt,
                                volume_integral::AbstractVolumeIntegral)
    nothing
end

function (limiter!::APosterioriLimiter)(u_ode, semi, t, dt,
                                        volume_integral::VolumeIntegralSubcellLimiting)
    @trixi_timeit timer() "a posteriori limiter" limiter!(u_ode, semi, t, dt,
                                                          volume_integral.indicator)
end

(::APosterioriLimiter)(u_ode, semi, t, dt, indicator::AbstractIndicator) = nothing

function (limiter!::APosterioriLimiter)(u_ode, semi, t, dt, indicator::IndicatorIDP)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    # Calculate blending factor alpha in [0,1]
    # f_ij = alpha_ij * f^(FV)_ij + (1 - alpha_ij) * f^(DG)_ij
    #      = f^(FV)_ij + (1 - alpha_ij) * f^(antidiffusive)_ij
    @trixi_timeit timer() "blending factors" solver.volume_integral.indicator(u, semi,
                                                                              solver, t,
                                                                              dt)

    perform_idp_correction!(u, dt, mesh, equations, solver, cache)

    return nothing
end

init_callback(limiter!::APosterioriLimiter, semi) = nothing

finalize_callback(limiter!::APosterioriLimiter, semi) = nothing

include("a_posteriori_limiter_2d.jl")
end # @muladd
