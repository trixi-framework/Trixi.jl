# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


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

  alive_callback = AliveCallback(0.0, alive_interval, analysis_interval)

  DiscreteCallback(alive_callback, alive_callback, # the first one is the condition, the second the affect!
                   save_positions=(false,false),
                   initialize=initialize!)
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AliveCallback})
  @nospecialize cb # reduce precompilation time

  alive_callback = cb.affect!
  print(io, "AliveCallback(alive_interval=", alive_callback.alive_interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:AliveCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    alive_callback = cb.affect!

    setup = [
             "interval" => alive_callback.alive_interval,
            ]
    summary_box(io, "AliveCallback", setup)
  end
end



function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AliveCallback}

  alive_callback = cb.affect!
  alive_callback.start_time = time_ns()
  return nothing
end


# this method is called to determine whether the callback should be activated
function (alive_callback::AliveCallback)(u, t, integrator)
  @unpack alive_interval, analysis_interval = alive_callback

  # With error-based step size control, some steps can be rejected. Thus,
  #   `integrator.iter >= integrator.stats.naccept`
  #    (total #steps)       (#accepted steps)
  # We need to check the number of accepted steps since callbacks are not
  # activated after a rejected step.
  return alive_interval > 0 && (
    (integrator.stats.naccept % alive_interval == 0 &&
    !(integrator.stats.naccept == 0 && integrator.iter > 0) &&
    (analysis_interval == 0 || integrator.stats.naccept % analysis_interval != 0)) ||
    isfinished(integrator))
end


# this method is called when the callback is activated
function (alive_callback::AliveCallback)(integrator)
  # Checking for floating point equality is OK here as `DifferentialEquations.jl`
  # sets the time exactly to the final time in the last iteration
  if isfinished(integrator) && mpi_isroot()
    println("─"^100)
    println("Trixi.jl simulation finished.  Final time: ", integrator.t,
            "  Time steps: ", integrator.stats.naccept, " (accepted), ", integrator.iter, " (total)")
    println("─"^100)
    println()
  elseif mpi_isroot()
    runtime_absolute = 1.0e-9 * (time_ns() - alive_callback.start_time)
    @printf("#timesteps: %6d │ Δt: %.4e │ sim. time: %.4e │ run time: %.4e s\n",
            integrator.stats.naccept, integrator.dt, integrator.t, runtime_absolute)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


end # @muladd
