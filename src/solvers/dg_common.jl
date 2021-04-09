function allocate_coefficients(mesh::Union{TreeMesh, CurvedMesh, UnstructuredQuadMesh}, equations, dg::DG, cache)
  # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
  # cf. wrap_array
  zeros(eltype(cache.elements), nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
end


@inline function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{1},CurvedMesh{1}}, equations, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
  # but that results in
  #   ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.

  # The following version is fast and allows us to `resize!(u_ode, ...)`.
  # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #      and other stuff that is only used indirectly via `wrap_array` afterwards!
  unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
              (nvariables(equations), nnodes(dg), nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::Union{TreeMesh{1},CurvedMesh{1}}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, element)
    end
  end
end


@inline function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{2},CurvedMesh{2},UnstructuredQuadMesh{2}}, equations, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
  # but that results in
  #   ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.

  # The following version is fast and allows us to `resize!(u_ode, ...)`.
  # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #      and other stuff that is only used indirectly via `wrap_array` afterwards!
  unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
              (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::Union{TreeMesh{2},CurvedMesh{2},UnstructuredQuadMesh{2}}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, element)
    end
  end
end


@inline function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{3},CurvedMesh{3}}, equations, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))
  # but that results in
  #   ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.

  # The following version is fast and allows us to `resize!(u_ode, ...)`.
  # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #      and other stuff that is only used indirectly via `wrap_array` afterwards!
  unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
              (nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::Union{TreeMesh{3},CurvedMesh{3}}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, k, element)
    end
  end
end
