
summary_callback(integrator) = false # when used as condition; never call the summary callback during the simulation
summary_callback(u, t, integrator) = u_modified!(integrator, false) # the summary callback does nothing when called accidentally


"""
    SummaryCallback()

Create and return a callback that prints a human-readable summary of the simulation setup at the
beginning of a simulation and then resets the timer. When the returned callback is executed
directly, the current timer values are shown.
"""
function SummaryCallback()
  DiscreteCallback(summary_callback, summary_callback,
                   save_positions=(false,false),
                   initialize=initialize_summary_callback)
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:typeof(summary_callback)}
  print(io, "SummaryCallback")
end


function format_key_value_line(key::AbstractString, value::AbstractString, key_width, total_width;
                               guide='…', filler='…', prefix="│ ", suffix=" │")
  @assert key_width < total_width
  line  = prefix
  squeezed_key = squeeze(key, key_width, filler=filler)
  line *= squeezed_key
  line *= ": "
  short = key_width - length(squeezed_key)
  if short <= 1
    line *= " "
  else
    line *= guide^(short-1) * " "
  end
  value_width = total_width - length(prefix) - length(suffix) - key_width - 2
  squeezed_value = squeeze(value, value_width, filler=filler)
  line *= squeezed_value
  short = value_width - length(squeezed_value)
  line *= " "^short
  line *= suffix

  @assert length(line) == total_width "should not happen: algorithm error!"

  return line
end
format_key_value_line(key, value, args...; kwargs...) = format_key_value_line(string(key), string(value), args...; kwargs...)

function squeeze(message, max_width; filler='…')
  @assert max_width >= 3 "squeezing works only for a minimum `max_width` of 3"

  length(message) <= max_width && return message

  keep_front = div(max_width, 2)
  keep_back = div(max_width, 2) - (isodd(max_width) ? 0 : 1)
  squeezed = message[begin:keep_front] * filler * message[end-keep_back+1:end]

  @assert length(squeezed) == max_width "should not happen: algorithm error!"

  return squeezed
end

function boxed_setup(heading, key_width, total_width, setup=[]; guide='…', filler='…')
  s  = ""
  s *= "┌" * "─"^(total_width-2) * "┐\n"
  s *= "│ " * heading * " "^(total_width - length(heading) - 4) * " │\n"
  # s *= "├" * "─"^(total_width-2) * "┤\n"
  s *= "│ " * "═"^length(heading) * " "^(total_width - length(heading) - 4) * " │\n"
  for (key, value) in setup
    s *= format_key_value_line(key, value, key_width, total_width) * "\n"
  end
  s *= "└" * "─"^(total_width-2) * "┘"
end


# Print information about the current simulation setup
# Note: This is called *after* all initialization is done, but *before* the first time step
function initialize_summary_callback(cb::DiscreteCallback, u, t, integrator)

  mpi_isroot() || return nothing

  print_startup_message()

  io = stdout
  key_width = 25
  total_width = 100 
  io_context = IOContext(io, :summary => true, :key_width => key_width, :total_width => total_width)

  semi = integrator.p
  show(io_context, MIME"text/plain"(), semi)
  println(io, "\n")
  mesh, equations, solver, _ = mesh_equations_solver_cache(semi)
  show(io_context, MIME"text/plain"(), mesh)
  println(io, "\n")
  show(io_context, MIME"text/plain"(), equations)
  println(io, "\n")
  show(io_context, MIME"text/plain"(), solver)
  println(io, "\n")

  callbacks = integrator.opts.callback
  if callbacks isa CallbackSet
    for cb in callbacks.continuous_callbacks
      show(io_context, MIME"text/plain"(), cb)
      println(io, "\n")
    end
    for cb in callbacks.discrete_callbacks
      # Do not show ourselves
      cb.affect! === summary_callback && continue

      show(io_context, MIME"text/plain"(), cb)
      println(io, "\n")
    end
  else
    show(io_context, MIME"text/plain"(), callbacks)
    println(io, "\n")
  end

  setup = Pair{String,Any}[
           "Start time" => first(integrator.sol.prob.tspan),
           "Final time" => last(integrator.sol.prob.tspan),
           "time integrator" => typeof(integrator.alg).name,
          ]
  println(io, boxed_setup("Time integration", key_width, total_width, setup))

  reset_timer!(timer())

  return nothing
end


function (cb::DiscreteCallback{Condition,Affect!})(io::IO=stdout) where {Condition, Affect!<:typeof(summary_callback)}

  mpi_isroot() || return nothing

  print_timer(io, timer(), title="Trixi.jl",
              allocations=true, linechars=:ascii, compact=false)
  println(io)
  return nothing
end
