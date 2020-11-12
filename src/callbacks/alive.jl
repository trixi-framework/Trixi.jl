
"""
    AliveCallback(analysis_interval=0, alive_interval=analysis_interval÷10)

Inexpensive callback showing that a simulation is still running by printing
some information such as the current time to the screen every `alive_interval`
time steps. If `analysis_interval ≂̸ 0`, the output is omitted every
`analysis_interval` time steps.
"""
mutable struct AliveCallback
  start_time::Float64
  alive_interval::Int
  analysis_interval::Int
end

function AliveCallback(; analysis_interval=0,
                         alive_interval=analysis_interval÷10)
  # when is the callback activated
  condition = (u, t, integrator) -> alive_interval > 0 && (
    (integrator.iter % alive_interval == 0 && (analysis_interval == 0 || integrator.iter % analysis_interval != 0)) ||
    isfinished(integrator))

  alive_callback = AliveCallback(0.0, alive_interval, analysis_interval)

  DiscreteCallback(condition, alive_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AliveCallback}
  alive_callback = cb.affect!
  print(io, "AliveCallback(alive_interval=", alive_callback.alive_interval, ")")
end
# TODO: Taal bikeshedding, implement a method with more information and the signature
# function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AliveCallback}
# end



function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AliveCallback}

  alive_callback = cb.affect!
  alive_callback.start_time = time_ns()
  return nothing
end


function (alive_callback::AliveCallback)(integrator)
  @unpack t, dt, iter = integrator

  # Checking for floating point equality is OK here as `DifferentialEquations.jl`
  # sets the time exactly to the final time in the last iteration
  if isfinished(integrator) && mpi_isroot()
    println("-"^80)
    println("Trixi simulation run finished.    Final time: ", integrator.t, "    Time steps: ", integrator.iter)
    println("-"^80)
    println()
  elseif mpi_isroot()
    runtime_absolute = 1.0e-9 * (time_ns() - alive_callback.start_time)
    @printf("#t/s: %6d | dt: %.4e | Sim. time: %.4e | Run time: %.4e s\n",
            iter, dt, t, runtime_absolute)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end
