# Tree mesh

The [`TreeMesh`](@ref) is a Cartesian, $h$-non-conforming mesh type
used in many parts of Trixi. Often, the support for this mesh type is
developed best since it was the first mesh type in Trixi,
and it is available in one, two, and three space dimensions.

It is limited to hypercube domains but supports AMR via the [`AMRCallback`](@ref).
Due to its Cartesian nature, (numerical) fluxes need to implement methods
dispatching on the `orientation::Integer` as described in the
[conventions](@ref conventions).
