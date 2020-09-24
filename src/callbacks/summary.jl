
function SummaryCallback()
  condition = (u, t, integrator) -> false

  affect! = (integrator) -> u_modified!(integrator, false)

  DiscreteCallback(condition, affect!,
                   save_positions=(false,false),
                   initialize=initialize_summary_callback)
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

  return nothing
end
