# Abstract supertype for DG-type solvers
# `POLYDEG` corresponds to `N` in the school of Kopriva
abstract type AbstractDg{NDIMS, POLYDEG, MeshType} <: AbstractSolver{NDIMS} end

@inline Base.ndims(dg::AbstractDg) = ndims(equations(dg))

# Return polynomial degree for a DG solver
@inline polydeg(::AbstractDg{NDIMS, POLYDEG}) where {NDIMS, POLYDEG} = POLYDEG

# Return number of nodes in one direction
@inline nnodes(::AbstractDg{NDIMS, POLYDEG}) where {NDIMS, POLYDEG} = POLYDEG + 1

# Return system of equations instance for a DG solver
@inline equations(dg::AbstractDg) = dg.equations

# Return number of variables for the system of equations in use
@inline nvariables(dg::AbstractDg) = nvariables(equations(dg))

# Return number of degrees of freedom
@inline ndofs(dg::AbstractDg) = dg.n_elements * nnodes(dg)^ndims(dg)

@inline uses_mpi(::AbstractDg{NDIMS, POLYDEG, TreeMesh{ParallelTree{NDIMS}}}) where {NDIMS, POLYDEG}= Val(true)
@inline uses_mpi(::AbstractDg{NDIMS, POLYDEG, TreeMesh{SerialTree{NDIMS}}}) where {NDIMS, POLYDEG} = Val(false)

"""
    get_node_coords(x, dg::AbstractDg, indices...)

Return an `ndims(dg)`-dimensional `SVector` for the DG node specified via the `i, j, k, element_id` indices (3D) or `i, j, element_id` indices (2D).
"""
@inline get_node_coords(x, dg::AbstractDg, indices...) = SVector(ntuple(idx -> x[idx, indices...], ndims(dg)))

"""
    get_node_vars(u, dg::AbstractDg, indices...)

Return an `nvariables(dg)`-dimensional `SVector` of the conservative variables for the DG node specified via the `i, j, k, element_id` indices (3D) or `i, j, element_id` indices (2D).
"""
@inline get_node_vars(u, dg::AbstractDg, indices...) = SVector(ntuple(v -> u[v, indices...], nvariables(dg)))

@inline function get_surface_node_vars(u, dg::AbstractDg, indices...)
  u_ll = SVector(ntuple(v -> u[1, v, indices...], nvariables(dg)))
  u_rr = SVector(ntuple(v -> u[2, v, indices...], nvariables(dg)))
  return u_ll, u_rr
end

@inline function set_node_vars!(u, u_node, ::AbstractDg, indices...)
  for v in eachindex(u_node)
    u[v, indices...] = u_node[v]
  end
  return nothing
end

@inline function add_to_node_vars!(u, u_node, ::AbstractDg, indices...)
  for v in eachindex(u_node)
    u[v, indices...] += u_node[v]
  end
  return nothing
end


# Include utilities
include("interpolation.jl")
include("l2projection.jl")

# Include 1D implementation
include("1d/containers.jl")
include("1d/dg.jl")
include("1d/amr.jl")

# Include 2D implementation
include("2d/containers.jl")
include("2d/dg.jl")
include("2d/amr.jl")
include("2d/parallel.jl")

# Include 3D implementation
include("3d/containers.jl")
include("3d/dg.jl")
include("3d/amr.jl")
