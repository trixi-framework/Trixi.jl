using Trixi

"""
    trixi2tec(u, semi, filename; title=basename(filename), solution_variables=cons2cons)

Convert a Trixi.jl solution array `u` for a given semidiscretization `semi` into a point-based
Tecplot ASCII file and store it as `filename`. Instead of manually extracting `u` and `semi`, you
can also just pass `sol` instead, i.e., the usual variable name used in Trixi.jl's elixirs for the
solution data.

You can optionally pass a `title` (by default the file portion of the filename is used). Further,
you can pass a different conversion function to `solution_variables` to write out variables in
non-conservative form.

### Example
```julia
julia> using Trixi

julia> trixi_include(default_example())

julia> trixi2tec(sol, "mydata.tec")

julia> trixi2tec(sol, "mydata_primitive.tec", solution_variables=cons2prim)
```

!!! warning "Experimental implementation"
    This only works for the `TreeMesh`, the `StructuredMesh`, the `UnstructuredMesh2D`,
    and the `P4estMesh`. In particular, it does not work for the `DGMulti` solver using the
    `DGMultiMesh`.

!!! warning "Experimental implementation"
    This is an experimental feature and *not* part of the official Trixi.jl API. Specifically,
    this function may change (or even be removed) in future releases without warning.
"""
function trixi2tec(u, semi, filename; title=basename(filename), solution_variables=cons2cons)
  # Extract fundamental building blocks and auxiliary data
  mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
  @unpack node_coordinates = cache.elements

  # Collect variable names and size information
  ndims = Trixi.ndims(semi)
  if ndims == 1
    variables = ["x"]
    ndofs_x = size(u, 2)
    indices = CartesianIndices((ndofs_x,))
    zone_info = "ZONE I=$ndofs_x, F=POINT\n"
  elseif ndims == 2
    variables = ["x", "y"]
    ndofs_x = size(u, 2)
    ndofs_y = size(u, 3)
    indices = CartesianIndices((ndofs_x, ndofs_y))
    zone_info = "ZONE I=$ndofs_x, J=$ndofs_y, F=POINT\n"
  elseif ndims == 3
    variables = ["x", "y", "z"]
    ndofs_x = size(u, 2)
    ndofs_y = size(u, 3)
    ndofs_z = size(u, 4)
    indices = CartesianIndices((ndofs_x, ndofs_y, ndofs_z))
    zone_info = "ZONE I=$ndofs_x, J=$ndofs_y, K=$ndofs_z, F=POINT\n"
  else
    error("Unsupported number of dimensions (must be 1, 2, or 3)")
  end
  push!(variables, Trixi.varnames(solution_variables, equations)...)
  variables_list = join(variables, "\", \"")

  # Write tec file
  open(filename, "w") do io
    write(io, """TITLE = "$title"\n""")
    write(io, """VARIABLES = "$variables_list"\n""")
    for element in eachelement(solver, cache)
      write(io, zone_info)
      for ci in indices
        node_coords = Trixi.get_node_coords(node_coordinates, equations, solver, ci, element)
        node_vars = solution_variables(Trixi.get_node_vars(u, equations, solver, ci, element), equations)
        print(io, join(node_coords, " "))
        write(io, " ")
        print(io, join(node_vars, " "))
        write(io, "\n")
      end # k, j, i
    end # element
  end
end

# Convenience function to allow calling `trixi2tec` with the `sol` variable
function trixi2tec(sol, filename; kwargs...)
  semi = sol.prob.p
  u_ode = sol.u[end]
  trixi2tec(u_ode, semi, filename; kwargs...)
end

# Convenience function to allow calling `trixi2tec` with, e.g., the initial condition
function trixi2tec(u_ode::Vector{<:Real}, semi, filename; kwargs...)
  u = Trixi.wrap_array_native(u_ode, semi)
  trixi2tec(u, semi, filename; kwargs...)
end
