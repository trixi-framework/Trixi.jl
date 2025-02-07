# Package extension for DiffEqBase.jl interface for Trixi.jl
module TrixiDiffEqBaseExt

using Trixi: AbstractTimeIntegrator
using DiffEqBase: DiffEqBase

DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator) = integrator.opts.tstops
DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator) = get_tstops(integrator).valtree
DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator) = maximum(get_tstops_array(integrator))

end # module TrixiDiffEqBaseExt
