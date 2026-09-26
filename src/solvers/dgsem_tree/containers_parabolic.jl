# Dimension-specific implementations

abstract type AbstractParabolicGradientBoundaryContainer end

include("container_parabolic_1d.jl")
include("container_parabolic_2d.jl")
include("container_parabolic_3d.jl")
