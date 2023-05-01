# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


summary_callback(u, t, integrator) = false # when used as condition; never call the summary callback during the simulation
summary_callback(integrator) = u_modified!(integrator, false) # the summary callback does nothing when called accidentally


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


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:typeof(summary_callback)})
  @nospecialize cb # reduce precompilation time

  print(io, "SummaryCallback")
end


# Format a key/value pair for output from the SummaryCallback
function format_key_value_line(key::AbstractString, value::AbstractString, key_width, total_width;
                               indentation_level=0, guide='…', filler='…', prefix="│ ", suffix=" │")
  @assert key_width < total_width
  line  = prefix
  # Indent the key as requested (or not at all if `indentation_level == 0`)
  indentation = prefix^indentation_level
  reduced_key_width = key_width - length(indentation)
  squeezed_key = indentation * squeeze(key, reduced_key_width, filler=filler)
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

# Squeeze a string to fit into a maximum width by deleting characters from the center
function squeeze(message, max_width; filler::Char='…')
  @assert max_width >= 3 "squeezing works only for a minimum `max_width` of 3"

  length(message) <= max_width && return message

  keep_front = div(max_width, 2)
  keep_back  = div(max_width, 2) - (isodd(max_width) ? 0 : 1)
  remove_back  = length(message) - keep_front
  remove_front = length(message) - keep_back
  squeezed = (chop(message, head=0, tail=remove_back)
              * filler *
              chop(message, head=remove_front, tail=0))

  @assert length(squeezed) == max_width "`$(length(squeezed)) != $max_width` should not happen: algorithm error!"

  return squeezed
end

# Print a summary with a box around it with a given heading and a setup of key=>value pairs
function summary_box(io::IO, heading, setup=[])
  summary_header(io, heading)
  for (key, value) in setup
    summary_line(io, key, value)
  end
  summary_footer(io)
end

function summary_header(io, heading; total_width=100, indentation_level=0)
  total_width = get(io, :total_width, total_width)
  indentation_level = get(io, :indentation_level, indentation_level)

  @assert indentation_level >= 0 "indentation level may not be negative"

  # If indentation level is greater than zero, we assume the header has already been printed
  indentation_level > 0 && return

  # Print header
  println(io, "┌" * "─"^(total_width-2) * "┐")
  println(io, "│ " * heading * " "^(total_width - length(heading) - 4) * " │")
  println(io, "│ " * "═"^length(heading) * " "^(total_width - length(heading) - 4) * " │")
end

function summary_line(io, key, value; key_width=30, total_width=100, indentation_level=0)
  # Printing is not performance-critical, so we can use `@nospecialize` to reduce latency
  @nospecialize value # reduce precompilation time

  key_width = get(io, :key_width, key_width)
  total_width = get(io, :total_width, total_width)
  indentation_level = get(io, :indentation_level, indentation_level)

  s = format_key_value_line(key, value, key_width, total_width,
                            indentation_level=indentation_level)

  println(io, s)
end

function summary_footer(io; total_width=100, indentation_level=0)
  total_width = get(io, :total_width, 100)
  indentation_level = get(io, :indentation_level, 0)

  if indentation_level == 0
    s = "└" * "─"^(total_width-2) * "┘"
  else
    s = ""
  end

  print(io, s)
end

@inline increment_indent(io) = IOContext(io, :indentation_level => get(io, :indentation_level, 0) + 1)


# Print information about the current simulation setup
# Note: This is called *after* all initialization is done, but *before* the first time step
function initialize_summary_callback(cb::DiscreteCallback, u, t, integrator)

  mpi_isroot() || return nothing

  print_startup_message()

  io = stdout
  io_context = IOContext(io,
                         :compact => false,
                         :key_width => 30,
                         :total_width => 100,
                         :indentation_level => 0)

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

  # time integration
  setup = Pair{String,Any}[
           "Start time" => first(integrator.sol.prob.tspan),
           "Final time" => last(integrator.sol.prob.tspan),
           "time integrator" => integrator.alg |> typeof |> nameof,
           "adaptive" => integrator.opts.adaptive,
          ]
  if integrator.opts.adaptive
    push!(setup,
      "abstol" => integrator.opts.abstol,
      "reltol" => integrator.opts.reltol,
      "controller" => integrator.opts.controller,
    )
  end
  summary_box(io, "Time integration", setup)
  println()

  # technical details
  setup = Pair{String,Any}[
           "#threads" => Threads.nthreads(),
          ]
  if mpi_isparallel()
    push!(setup,
      "#MPI ranks" => mpi_nranks(),
    )
  end
  summary_box(io, "Environment information", setup)
  println()

  reset_timer!(timer())

  return nothing
end


function (cb::DiscreteCallback{Condition,Affect!})(io::IO=stdout) where {Condition, Affect!<:typeof(summary_callback)}

  mpi_isroot() || return nothing

  TimerOutputs.complement!(timer())
  print_timer(io, timer(), title="Trixi.jl",
              allocations=true, linechars=:unicode, compact=false)
  println(io)
  return nothing
end


end # @muladd
