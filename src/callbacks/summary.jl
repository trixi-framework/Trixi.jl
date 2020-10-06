
summary_callback(integrator) = t == false
summary_callback(u, t, integrator) = u_modified!(integrator, false)

function SummaryCallback()
  DiscreteCallback(summary_callback, summary_callback,
                   save_positions=(false,false),
                   initialize=initialize_summary_callback)
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:typeof(summary_callback)}
  print(io, "SummaryCallback")
end


function initialize_summary_callback(cb::DiscreteCallback, u, t, integrator)

  print_startup_message()

  io = stdout
  semi = integrator.p
  show(io, MIME"text/plain"(), semi)
  println(io, "\n")
  mesh, equations, solver, _ = mesh_equations_solver_cache(semi)
  show(io, MIME"text/plain"(), mesh)
  println(io, "\n")
  show(io, MIME"text/plain"(), equations)
  println(io, "\n")
  show(io, MIME"text/plain"(), solver)
  println(io, "\n")

  callbacks = integrator.opts.callback
  if callbacks isa CallbackSet
    for cb in callbacks.continuous_callbacks
      show(io, MIME"text/plain"(), cb)
      println(io, "\n")
    end
    for cb in callbacks.discrete_callbacks
      show(io, MIME"text/plain"(), cb)
      println(io, "\n")
    end
  end

  # TODO: Taal decide, shall we print more information about the ODE problem/algorithm?
  println(io, "tspan = ", integrator.sol.prob.tspan)
  println("Time integrator: ", integrator.alg)

  reset_timer!(timer())

  return nothing
end


function (cb::DiscreteCallback{Condition,Affect!})(io::IO=stdout) where {Condition, Affect!<:typeof(summary_callback)}
  print_timer(io, timer(), title="Trixi.jl",
              allocations=true, linechars=:ascii, compact=false)
  println(io)
end
