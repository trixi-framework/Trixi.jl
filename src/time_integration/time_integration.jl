# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Wrapper type for solutions from Trixi.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

# Abstract supertype of Trixi.jl's own time integrators for dispatch
abstract type AbstractTimeIntegrator end

# Interface required by DiffEqCallbacks.jl
function DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator)
    return integrator.opts.tstops
end
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
    return get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
    return maximum(get_tstops_array(integrator))
end

include("methods_2N.jl")
include("methods_3Sstar.jl")
include("methods_SSP.jl")
include("paired_explicit_runge_kutta/paired_explicit_runge_kutta.jl")

end # @muladd
