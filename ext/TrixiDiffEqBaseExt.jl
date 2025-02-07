# Package extension for DiffEqBase.jl interface for Trixi.jl
module TrixiDiffEqBaseExt

using Trixi: AbstractTimeIntegrator
using DiffEqBase: DiffEqBase

DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator) = integrator.opts.tstops
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
    get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
    maximum(get_tstops_array(integrator))
end

end # module TrixiDiffEqBaseExt
