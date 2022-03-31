# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Default value - meshes supporting parallelization via MPI must opt-in
mpi_parallel(mesh::AbstractMesh) = Val{false}()

include("tree_mesh.jl")
include("structured_mesh.jl")
include("surface_interpolant.jl")
include("unstructured_mesh.jl")
include("face_interpolant.jl")
include("transfinite_mappings_3d.jl")
include("p4est_mesh.jl")
include("mesh_io.jl")
include("dgmulti_meshes.jl")


end # @muladd
