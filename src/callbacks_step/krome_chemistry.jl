# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


  """
      KROMEChemistryCallback()
  Apply Chemistry with KROME Package
  """
  function KROMEChemistryCallback()
    DiscreteCallback(krome_chemistry_callback, krome_chemistry_callback,
                     save_positions=(false,false),
                     initialize=initialize!)
  end
  
  # Always execute collision step after a time step, but not after the last step
  krome_chemistry_callback(u, t, integrator) = !isfinished(integrator)
  
  
  function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:typeof(krome_chemistry_callback)})
    @nospecialize cb # reduce precompilation time
  
    print(io, "KROMEChemistryCallback()")
  end
  
  
  function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:typeof(krome_chemistry_callback)})
    @nospecialize cb # reduce precompilation time
  
    if get(io, :compact, false)
      show(io, cb)
    else
      summary_box(io, "KROMEChemistryCallback")
    end
  end
  
  
  # Execute collision step once in the very beginning
  function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:typeof(krome_chemistry_callback)}
    cb.affect!(integrator)
  end
  
  
  # This method is called as callback after the StepsizeCallback during the time integration.
  @inline function krome_chemistry_callback(integrator)
  
    dt = get_proposed_dt(integrator)
    semi = integrator.p
    #mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    mesh = semi.mesh
    equations = semi.equations
    solver = semi.solver
    cache = semi.cache
    chemistry_terms = semi.chemistry_terms
  
    u_ode = integrator.u
    u = wrap_array(u_ode, mesh, equations, solver, cache)
  
    @trixi_timeit timer() "KROME Chemistry" apply_krome_chemistry!(u, dt, mesh, chemistry_terms, equations, solver, cache)
  
    return nothing
  end
  
  include("krome_chemistry_dg1d.jl")
  include("krome_chemistry_dg2d.jl")
  
  
  end # @muladd