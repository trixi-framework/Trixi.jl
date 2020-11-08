
"""
    TrivialCallback()

A callback that does nothing. This can be useful to disable some callbacks
easily via [`trixi_include`](@ref).
"""
function TrivialCallback()
  DiscreteCallback(trivial_callback, trivial_callback,
                   save_positions=(false,false))
end

trivial_callback(u, t, integrator) = false
trivial_callback(integrator) = u_modified!(integrator, false)


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:typeof(trivial_callback)}
  alive_callback = cb.affect!
  print(io, "TrivialCallback()")
end

