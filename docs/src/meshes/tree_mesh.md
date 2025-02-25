# Tree mesh

The [`TreeMesh`](@ref) is a Cartesian, $h$-non-conforming mesh type
used in many parts of Trixi.jl. Often, the support for this mesh type is
developed best since it was the first mesh type in Trixi.jl,
and it is available in one, two, and three space dimensions.

It is limited to hypercube domains (that is, lines in 1D, squares in 2D and cubes in 3D) but supports AMR via the [`AMRCallback`](@ref).
Due to its Cartesian nature, (numerical) fluxes need to implement methods
dispatching on the `orientation::Integer` as described in the
[conventions](@ref conventions).
