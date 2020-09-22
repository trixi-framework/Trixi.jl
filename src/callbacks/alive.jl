
mutable struct AliveCallback
  start_time::Float64
end

function (alive_callback::AliveCallback)(integrator)
  if integrator.t == integrator.sol.prob.tspan[2]
    println("-"^80)
    println("Trixi simulation run finished.    Final time: ", integrator.t, "    Time steps: ", integrator.iter)
    println("-"^80)
    println()

    print_timer(timer(), title="Trixi.jl",
                allocations=true, linechars=:ascii, compact=false)
    println()
  else
    @unpack t, dt, iter = integrator
    runtime_absolute = 1.0e-9 * (time_ns() - alive_callback.start_time)
    @printf("#t/s: %6d | dt: %.4e | Sim. time: %.4e | Run time: %.4e s\n",
            iter, dt, t, runtime_absolute)
  end

  return nothing
end

function AliveCallback(; analysis_interval=0,
                         alive_interval=analysis_interval÷10)
  condition = (u, t, integrator) -> alive_interval > 0 && ((integrator.iter % alive_interval == 0 && integrator.iter % analysis_interval != 0) || t == integrator.sol.prob.tspan[2])

  alive_callback = AliveCallback(0.0)

  DiscreteCallback(condition, alive_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end

function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AliveCallback}
  reset_timer!(timer())
  alive_callback = cb.affect!
  alive_callback.start_time = time_ns()
  return nothing
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AliveCallback}
  stepsize_callback = cb.affect!
  print(io, "AliveCallback")
end
# TODO: Taal bikeshedding, implement a method with more information and the signature
# function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:StepsizeCallback}
# end
