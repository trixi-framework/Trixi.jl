# Abstract supertype for DG-type solvers
# `POLYDEG` corresponds to `N` in the school of Kopriva
abstract type AbstractDg{NDIMS, POLYDEG} <: AbstractSolver{NDIMS} end

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

@inline get_node_coords(x, dg::AbstractDg, indices...) = SVector(ntuple(idx -> x[idx, indices...], ndims(dg)))

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

# Include 2D implementation
include("2d/containers.jl")
include("2d/dg.jl")
include("2d/amr.jl")

# Include 3D implementation
include("3d/containers.jl")
include("3d/dg.jl")
include("3d/amr.jl")
