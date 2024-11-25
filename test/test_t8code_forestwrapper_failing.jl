module TestT8codeMeshAutomaticCleanUpFailing

using Test
using Trixi

include("test_trixi.jl")

@trixi_testset "test T8codeMesh failing automatic cleanup" begin
    comm = Trixi.mpi_comm()

    # Create a forest and do not wrap it by `ForestWrapper`.
    scheme = Trixi.t8_scheme_new_default_cxx()
    cmesh = Trixi.t8_cmesh_new_hypercube(Trixi.T8_ECLASS_QUAD, comm, 0, 0, 0)
    forest = Trixi.t8_forest_new_uniform(cmesh, scheme, 0, 0, comm)

    # No wrapper object registered.
    @test length(Trixi.T8code.T8CODE_OBJECT_TRACKER) == 0

    # We expect that it throws the warning below and `libsc` prints 'Memory imbalance' to stderr.
    @test_warn "Inconsistent state detected after finalizing t8code." Trixi.MPI.Finalize()===Nothing
end

end # module
