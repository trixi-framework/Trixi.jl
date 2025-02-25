module TestExamplesMPIP4estMesh3D

using Test
using Trixi

include("test_trixi.jl")

const EXAMPLES_DIR = pkgdir(Trixi, "examples", "p4est_3d_dgsem")

@testset "P4estMesh MPI 3D" begin
#! format: noindent

# Run basic tests
@testset "Examples 3D" begin
    # Linear scalar advection
    @trixi_testset "elixir_advection_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[0.00016263963870641478],
                            linf=[0.0014537194925779984])

        @testset "error-based step size control" begin
            Trixi.mpi_isroot() && println("-"^100)
            Trixi.mpi_isroot() &&
                println("elixir_advection_basic.jl with error-based step size control")

            sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                        ode_default_options()..., callback = callbacks)
            summary_callback()
            errors = analysis_callback(sol)
            if Trixi.mpi_isroot()
                @test errors.l2≈[0.00016800412839949264] rtol=1.0e-4
                @test errors.linf≈[0.0014548839020096516] rtol=1.0e-4
            end
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[9.773852895157622e-6],
                            linf=[0.0005853874124926162],)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_unstructured_curved.jl"),
                            l2=[1.6163120948209677e-5],
                            linf=[0.0010572201890564834],
                            tspan=(0.0, 1.0),)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_restart.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                            l2=[0.002590388934758452],
                            linf=[0.01840757696885409],)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_cubed_sphere.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
                            l2=[0.002006918015656413],
                            linf=[0.027655117058380085])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # Compressible Euler
    @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
                            l2=[
                                4.070355207909268e-5,
                                4.4993257426833716e-5,
                                5.10588457841744e-5,
                                5.102840924036687e-5,
                                0.00019986264001630542
                            ],
                            linf=[
                                0.0016987332417202072,
                                0.003622956808262634,
                                0.002029576258317789,
                                0.0024206977281964193,
                                0.008526972236273522
                            ],
                            tspan=(0.0, 0.01))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_nonperiodic.jl"),
                            l2=[
                                0.0015106060984283647,
                                0.0014733349038567685,
                                0.00147333490385685,
                                0.001473334903856929,
                                0.0028149479453087093
                            ],
                            linf=[
                                0.008070806335238156,
                                0.009007245083113125,
                                0.009007245083121784,
                                0.009007245083102688,
                                0.01562861968368434
                            ],
                            tspan=(0.0, 1.0))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                            l2=[
                                0.010380390326164493,
                                0.006192950051354618,
                                0.005970674274073704,
                                0.005965831290564327,
                                0.02628875593094754
                            ],
                            linf=[
                                0.3326911600075694,
                                0.2824952141320467,
                                0.41401037398065543,
                                0.45574161423218573,
                                0.8099577682187109
                            ],
                            tspan=(0.0, 0.2),)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_source_terms_nonperiodic_hohqmesh.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_nonperiodic_hohqmesh.jl"),
                            l2=[
                                0.0042023406458005464,
                                0.004122532789279737,
                                0.0042448149597303616,
                                0.0036361316700401765,
                                0.007389845952982495
                            ],
                            linf=[
                                0.04530610539892499,
                                0.02765695110527666,
                                0.05670295599308606,
                                0.048396544302230504,
                                0.1154589758186293
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_mhd_alfven_wave_nonconforming.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_mhd_alfven_wave_nonconforming.jl"),
                            l2=[
                                0.0001788543743594658,
                                0.000624334205581902,
                                0.00022892869974368887,
                                0.0007223464581156573,
                                0.0006651366626523314,
                                0.0006287275014743352,
                                0.000344484339916008,
                                0.0007179788287557142,
                                8.632896980651243e-7
                            ],
                            linf=[
                                0.0010730565632763867,
                                0.004596749809344033,
                                0.0013235269262853733,
                                0.00468874234888117,
                                0.004719267084104306,
                                0.004228339352211896,
                                0.0037503625505571625,
                                0.005104176909383168,
                                9.738081186490818e-6
                            ],
                            tspan=(0.0, 0.25),)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # Same test as above but with only one tree in the mesh
    # We use it to test meshes with elements of different size in each partition
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.0019054118500017054,
                            0.006957977226608083,
                            0.003429930594167365,
                            0.009051598556176287,
                            0.0077261662742688425,
                            0.008210851821439208,
                            0.003763030674412298,
                            0.009175470744760567,
                            2.881690753923244e-5
                        ],
                        linf=[
                            0.010983704624623503,
                            0.04584128974425262,
                            0.02022630484954286,
                            0.04851342295826149,
                            0.040710154751363525,
                            0.044722299260292586,
                            0.036591209423654236,
                            0.05701669133068068,
                            0.00024182906501186622
                        ],
                        tspan=(0.0, 0.25), trees_per_dimension=(1, 1, 1),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end # P4estMesh MPI

end # module
