# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct EntropyBoundedLimiter{RealT <: Real}
    exp_entropy_decrease_max::RealT # < 0
end

"""
    EntropyBoundedLimiter{RealT <: Real}(; exp_entropy_decrease_max = 1f-13)

Entropy-bounded limiter by
- Lv, Ihme (2015)
  Entropy-bounded discontinuous Galerkin scheme for Euler equations
  [doi: 10.1016/j.jcp.2015.04.026](https://doi.org/10.1016/j.jcp.2015.04.026)

This is an ideal-gas specific limiter that bounds the (unphysical) decrease of the thermodynamic entropy 
per element from one time step (or Runge-Kutta stage) to the next.
The parameter `exp_entropy_decrease_max` is the maximum allowed exponentiated
entropy decrease per element at each element's node.

In the original version of the paper, this value is set to zero to
ensure that the entropy does not decrease, i.e., guarantee entropy stability in the sense of 
- Tadmor (1986)
  A minimum entropy principle in the gas dynamics equations
  [doi: 10.1016/0168-9274(86)90029-2](https://doi.org/10.1016/0168-9274(86)90029-2)

This, however, leads in general to very diffusive solutions for timesteps violating 
a CFL condition (Lemma 3 in Lv, Ihme (2015)) which is required for entropy stability in the mean values.
Since most practical simulations will employ a significantly larger timestep, one can relax the 
strict entropy increase requirement by setting `exp_entropy_decrease_max` to a negative value.
The limiter acts if the exponentiated entropy decrease on an element is larger than
`exp_entropy_decrease_max`.
This means that if the change in exponentiated entropy lies *below* `exp_entropy_decrease_max` (i.e., larger in absolute value) the limiter takes action.
The choice of the tolerated exponentiated entropy decrease is a problem-specific parameter 
which balances the trade-off between accuracy and stability.
"""
function EntropyBoundedLimiter(;
                               exp_entropy_decrease_max::RealT = -1.0f-13) where {RealT <:
                                                                                  Real}
    @assert exp_entropy_decrease_max<0 "Supplied `exp_entropy_decrease_max` expected to be negative"
    EntropyBoundedLimiter{RealT}(exp_entropy_decrease_max)
end

function (limiter!::EntropyBoundedLimiter)(u_ode, integrator,
                                           semi::AbstractSemidiscretization,
                                           t)
    @trixi_timeit timer() "entropy-bounded limiter" begin
        @assert :uprev in fieldnames(typeof(integrator)) "EntropyBoundedLimiter requires `uprev` for computation of previous entropy"

        u = wrap_array(u_ode, semi)
        u_prev = wrap_array(integrator.uprev, semi)
        limiter_entropy_bounded!(u, u_prev, limiter!.exp_entropy_decrease_max,
                                 mesh_equations_solver_cache(semi)...)
    end
end

# Exponentiated entropy change for the thermodynamic entropy (see `entropy_thermodynamic`) 
# of an ideal gas with constant gamma.
@inline function exp_entropy_change(p, rho, gamma, exp_entropy_prev)
    return p - rho^gamma * exp_entropy_prev
end

include("entropy_bounded_limiter_1d.jl")
include("entropy_bounded_limiter_2d.jl")
include("entropy_bounded_limiter_3d.jl")
end # @muladd
