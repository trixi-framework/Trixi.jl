module TestT8codeMeshAutomaticCleanUp

using Test
using Trixi

include("test_trixi.jl")

# Dummy variable in order to supress a warning.
EXAMPLES_DIR = joinpath(examples_dir(), "t8code_2d_dgsem")

@trixi_testset "test T8codeMesh automatic cleanup" begin
    @test length(Trixi.T8code.T8CODE_OBJECT_TRACKER) == 0

    comm = Trixi.mpi_comm()

    # Create a forest and wrap by `ForestWrapper`
    scheme = Trixi.t8_scheme_new_default_cxx()
    cmesh = Trixi.t8_cmesh_new_hypercube(Trixi.T8_ECLASS_QUAD, comm, 0, 0, 0)
    forest = Trixi.t8_forest_new_uniform(cmesh, scheme, 0, 0, comm)
    wrapper_A = Trixi.T8code.ForestWrapper(forest)

    @test length(Trixi.T8code.T8CODE_OBJECT_TRACKER) == 1

    # Create another forest and wrap by `ForestWrapper`
    scheme = Trixi.t8_scheme_new_default_cxx()
    cmesh = Trixi.t8_cmesh_new_hypercube(Trixi.T8_ECLASS_TRIANGLE, comm, 0, 0, 0)
    forest = Trixi.t8_forest_new_uniform(cmesh, scheme, 0, 0, comm)
    wrapper_B = Trixi.T8code.ForestWrapper(forest)

    @test length(Trixi.T8code.T8CODE_OBJECT_TRACKER) == 2

    # Finalize the first wrapper.
    finalize(wrapper_A)

    @test length(Trixi.T8code.T8CODE_OBJECT_TRACKER) == 1

    # The second wrapper should be finalized automatically when Julia shuts down.
    # ... finalize(wrapper_B) ...

    @test_nowarn Trixi.MPI.Finalize() === Nothing
end

end # module
