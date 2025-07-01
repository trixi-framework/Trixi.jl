abstract type RelaxationIntegrator <: AbstractTimeIntegrator end

get_tmp_cache(integrator::RelaxationIntegrator) = (integrator.u_tmp,)

# stop the time integration
function terminate!(integrator::RelaxationIntegrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

@inline function update_t_relaxation!(integrator::RelaxationIntegrator)
    # Check if due to entropy relaxation the final time would not be reached
    if integrator.finalstep == true && integrator.gamma != 1
        integrator.gamma = 1.0
    end
    integrator.t += integrator.gamma * integrator.dt

    return nothing
end

include("entropy_relaxation.jl")
include("methods_subdiagonal.jl")
include("methods_vanderHouwen.jl")
