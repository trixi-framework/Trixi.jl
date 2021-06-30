# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function allocate_coefficients(mesh::AbstractMesh, equations, dg::DG, cache)
  # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
  # cf. wrap_array
  zeros(eltype(cache.elements), nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
end

@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations, dg::DGSEM, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #     reshape(u_ode, (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
  # but that results in
  #     ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.
  #
  # !!! danger "Segfaults"
  #     Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #     and other stuff that is only used indirectly via `wrap_array` afterwards!

  # Currently, there are problems when AD is used with `PtrArray`s in broadcasts
  # since LoopVectorization does not support `ForwardDiff.Dual`s. Hence, we use
  # optimized `PtrArray`s whenever possible and fall back to plain `Array`s
  # otherwise.
  if LoopVectorization.check_args(u_ode)
    # This version using `PtrArray`s from StrideArrays.jl is very fast and
    # does not result in allocations.
    #
    # !!! danger "Heisenbug"
    #     Do not use this code when `@threaded` uses `Threads.@threads`. There is
    #     a very strange Heisenbug that makes some parts very slow *sometimes*.
    #     In fact, everything can be fast and fine for many cases but some parts
    #     of the RHS evaluation can take *exactly* (!) five seconds randomly...
    #     Hence, this version should only be used when `@threaded` is based on
    #     `@batch` from Polyester.jl or something similar. Using Polyester.jl
    #     is probably the best option since everything will be handed over to
    #     Chris Elrod, one of the best performance software engineers for Julia.
    PtrArray(pointer(u_ode),
             (StaticInt(nvariables(equations)), ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))..., nelements(dg, cache)))
            #  (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
  else
    # The following version is reasonably fast and allows us to `resize!(u_ode, ...)`.
    unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
                (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
  end
end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations, dg::DG, cache)
  wrap_array_native(u_ode, mesh, equations, dg, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
              (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::AbstractMesh{1}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, element)
    end
  end
end


function compute_coefficients!(u, func, t, mesh::AbstractMesh{2}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, element)
    end
  end
end


function compute_coefficients!(u, func, t, mesh::AbstractMesh{3}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, k, element)
    end
  end
end


end # @muladd
