# Tree mesh

The [`TreeMesh`](@ref) is a Cartesian, possibly $h$-non-conforming mesh type
used in most parts of Trixi.
It is limited to hypercube domains but supports AMR via the [`AMRCallback`](@ref).
Due to its Cartesian nature, (numerical) fluxes need to implement methods
dispatching on the `orientation::Integer` as described in the
[conventions](@ref conventions).
