module TestExamplesMPIP4estMesh2DParabolic

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

@testset "P4estMesh MPI 3D Parabolic" begin
    @trixi_testset "P4estMesh3D: elixir_navierstokes_taylor_green_vortex_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_taylor_green_vortex_amr.jl"),
                            initial_refinement_level=0,
                            max_level=2,
                            tspan=(0.0, 0.1),
                            l2=[
                                0.0011069115461970517,
                                0.013872454764036899,
                                0.013872454764036934,
                                0.012060120516483785,
                                0.14491993697252206
                            ],
                            linf=[
                                0.004408900543641403,
                                0.05154019471576565,
                                0.051540194715650245,
                                0.035283556918085636,
                                0.6804810816393854
                            ])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh3D: elixir_navierstokes_taylor_green_vortex_amr_mortar.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_taylor_green_vortex_amr_mortar.jl"),
                            tspan=(0.0, 5.0),
                            l2=[
                                0.001716903538135598,
                                0.2566781540081305,
                                0.2566781540081202,
                                0.22556320347015937,
                                0.33779107381452556
                            ],
                            linf=[
                                0.01139265672066836,
                                1.4145605813055888,
                                1.4145605813076223,
                                1.070081196321056,
                                3.332907472310012
                            ])
    end
    @trixi_testset "P4estMesh3D: elixir_navierstokes_blast_wave_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_blast_wave_amr.jl"),
                            tspan=(0.0, 0.01),
                            l2=[
                                0.009449115832266491,
                                0.0017932092857965453,
                                0.0017932092857965449,
                                0.001793209285796548,
                                0.02432855189940458
                            ],
                            linf=[
                                0.6811440777026873,
                                0.17744074602770776,
                                0.17744074602770762,
                                0.1774407460277074,
                                1.7402299022804495
                            ])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end
    @trixi_testset "P4estMesh3D: elixir_advection_diffusion_nonperiodic.jl (LDG)" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_diffusion_nonperiodic.jl"),
                            solver_parabolic=ParabolicFormulationLocalDG(),
                            cfl_parabolic=0.07,
                            l2=[0.004185076476662267], linf=[0.05166349548111486])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh3D: elixir_advection_diffusion_amr_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_diffusion_amr_curved.jl"),
                            l2=[0.000683123952524889], linf=[0.023601069354373894])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh3D: elixir_advection_diffusion_amr_curved.jl (LDG)" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_diffusion_amr_curved.jl"),
                            solver_parabolic=ParabolicFormulationLocalDG(),
                            l2=[0.0006853004145232737], linf=[0.02352694543085776])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh3D: elixir_navierstokes_freestream_boundaries.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_freestream_boundaries.jl"),
                            tspan=(0.0, 0.1),
                            l2=[
                                1.050376383380673e-16,
                                1.0175313793753473e-16,
                                1.158489273890016e-16,
                                2.0654608507933775e-16,
                                3.3590256030698164e-15
                            ],
                            linf=[
                                1.7763568394002505e-15,
                                1.0130785099704553e-15,
                                1.3322676295501878e-15,
                                2.4424906541753444e-15,
                                4.263256414560601e-14
                            ])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end
end #Testset
end # module
