# Abstract supertype for DG-type solvers
abstract type AbstractDg{NDIMS} <: AbstractSolver{NDIMS} end

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
