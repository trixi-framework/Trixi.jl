# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    TrivialCallback()

A callback that does nothing. This can be useful to disable some callbacks easily via
[`trixi_include`](@ref).
"""
function TrivialCallback()
    DiscreteCallback(trivial_callback, trivial_callback,
                     save_positions = (false, false))
end

trivial_callback(u, t, integrator) = false
trivial_callback(integrator) = u_modified!(integrator, false)

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:typeof(trivial_callback)})
    @nospecialize cb # reduce precompilation time

    print(io, "TrivialCallback()")
end

# This allows to set `summary_callback = TrivialCallback()` in elixirs to suppress
# output, e.g. in `convergence_test`.
function (cb::DiscreteCallback{Condition, Affect!})(io::IO = stdout) where {Condition,
                                                                            Affect! <:
                                                                            typeof(trivial_callback)
                                                                            }
    return nothing
end
end # @muladd
