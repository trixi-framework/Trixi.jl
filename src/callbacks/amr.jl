
# TODO: Taal design, how can we implement AMR as callbacks? Xref https://github.com/SciML/OrdinaryDiffEq.jl/pull/1275
# Currently, the best option seems to be to let OrdinaryDiffEq.jl use `Vector`s,
# which can be `resize!`ed for AMR. Then, we have to wrap these `Vector`s inside
# Trixi.jl as our favorite multidimensional array type along the lines of
# unsafe_wrap(Array{eltype(u), ndims(mesh)+2}, pointer(u), (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache))
# in the two-dimensional case. We would need to do this wrapping in every
# method exposed to OrdinaryDiffEq, i.e. in the first levels of things like
# rhs!, AMRCallback, StepsizeCallback, AnalysisCallback, SaveSolutionCallback
mutable struct AMRCallback{RealT, Indicator, Cache}
  interval::Int
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  indicator::Indicator
  cache::Cache
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   print(io, "AMRCallback")
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  @unpack interval, alpha_max, alpha_min, alpha_smooth, indicator = amr_callback
  println(io, "AMRCallback with")
  println(io, "- interval:     ", interval)
  println(io, "- alpha_max:    ", alpha_max)
  println(io, "- alpha_min:    ", alpha_min)
  println(io, "- alpha_smooth: ", alpha_smooth)
  print(io,   "- indicator:    ", indicator)
end


function AMRCallback(indicator; interval=5,
                                alpha_max=1.0, alpha_min=0.0, alpha_smooth=false)
  condition = (u, t, integrator) -> interval > 0 && (integrator.iter % interval == 0)

  # TODO: Taal, implement
  error("TODO")
  amr_callback = AMRCallback(0.0)

  DiscreteCallback(condition, amr_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AMRCallback}
  reset_timer!(timer())
  amr_callback = cb.affect!
  # TODO: Taal, implement
  return nothing
end


function (amr_callback::AMRCallback)(integrator)
  # TODO: Taal, implement

  return nothing
end
