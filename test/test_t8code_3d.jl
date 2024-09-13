module TestExamplesT8codeMesh3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "t8code_3d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)
mkdir(outdir)

@testset "T8codeMesh3D" begin
    @trixi_testset "test t8code mesh from p8est connectivity" begin
        @test begin
            # Here we use the connectivity constructor from `P4est.jl` since the
            # method dispatch works only on `Ptr{p8est_connectivity}` which
            # actually is `Ptr{P4est.LibP4est.p8est_connectivity}`.
            conn = Trixi.P4est.LibP4est.p8est_connectivity_new_brick(2, 3, 4, 1, 1, 1)
            mesh = T8codeMesh(conn)
            all(size(mesh.tree_node_coordinates) .== (3, 2, 2, 2, 24))
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl`.
    @trixi_testset "elixir_advection_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[0.00016263963870641478],
                            linf=[0.0014537194925779984])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl`.
    @trixi_testset "elixir_advection_unstructured_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_unstructured_curved.jl"),
                            l2=[0.0004750004258546538],
                            linf=[0.026527551737137167])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl`.
    @trixi_testset "elixir_advection_nonconforming.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
                            l2=[0.00253595715323843],
                            linf=[0.016486952252155795])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl` besides minor
    # deviations from the expected error norms.
    @trixi_testset "elixir_advection_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[1.1302812803902801e-5],
                            linf=[0.0007889950196294793],
                            coverage_override=(maxiters = 6, initial_refinement_level = 1,
                                               base_level = 1, med_level = 2, max_level = 3))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl` besides minor
    # deviations from the expected error norms.
    @trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_unstructured_curved.jl"),
                            l2=[2.0535121347526814e-5],
                            linf=[0.0010586603797777504],
                            tspan=(0.0, 1.0),
                            coverage_override=(maxiters = 6, initial_refinement_level = 0,
                                               base_level = 0, med_level = 1, max_level = 2))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl`.
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

    # This test is identical to the one in `test_p4est_3d.jl`.
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

    # This test is identical to the one in `test_p4est_3d.jl`.
    @trixi_testset "elixir_euler_free_stream.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                            l2=[
                                5.162664597942288e-15,
                                1.941857343642486e-14,
                                2.0232366394187278e-14,
                                2.3381518645408552e-14,
                                7.083114561232324e-14
                            ],
                            linf=[
                                7.269740365245525e-13,
                                3.289868377720495e-12,
                                4.440087186807773e-12,
                                3.8686831516088205e-12,
                                9.412914891981927e-12
                            ],
                            tspan=(0.0, 0.03))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl`.
    @trixi_testset "elixir_euler_free_stream_extruded.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_extruded.jl"),
                            l2=[
                                8.444868392439035e-16,
                                4.889826056731442e-15,
                                2.2921260987087585e-15,
                                4.268460455702414e-15,
                                1.1356712092620279e-14
                            ],
                            linf=[
                                7.749356711883593e-14,
                                2.8792246364872653e-13,
                                1.1121659149182506e-13,
                                3.3228975127030935e-13,
                                9.592326932761353e-13
                            ],
                            tspan=(0.0, 0.1), atol=5.0e-13,)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl`.
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
                            tspan=(0.0, 0.2),
                            coverage_override=(polydeg = 3,)) # Prevent long compile time in CI
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # This test is identical to the one in `test_p4est_3d.jl` besides minor
    # deviations in the expected error norms.
    @trixi_testset "elixir_euler_sedov.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                            l2=[
                                7.82070951e-02,
                                4.33260474e-02,
                                4.33260474e-02,
                                4.33260474e-02,
                                3.75260911e-01
                            ],
                            linf=[
                                7.45329845e-01,
                                3.21754792e-01,
                                3.21754792e-01,
                                3.21754792e-01,
                                4.76151527e+00
                            ],
                            tspan=(0.0, 0.3),
                            coverage_override=(polydeg = 3,)) # Prevent long compile time in CI
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
        @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "tree_3d_dgsem"),
                                     "elixir_euler_convergence_pure_fv.jl"),
                            l2=[
                                0.037182410351406,
                                0.032062252638283974,
                                0.032062252638283974,
                                0.03206225263828395,
                                0.12228177813586687
                            ],
                            linf=[
                                0.0693648413632646,
                                0.0622101894740843,
                                0.06221018947408474,
                                0.062210189474084965,
                                0.24196451799555962
                            ],
                            mesh=T8codeMesh((4, 4, 4), polydeg = 3,
                                            coordinates_min = (0.0, 0.0, 0.0),
                                            coordinates_max = (2.0, 2.0, 2.0)),
                            # Remove SaveSolution callback
                            callbacks=CallbackSet(summary_callback,
                                                  analysis_callback, alive_callback,
                                                  stepsize_callback))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_weak_blast_wave_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_amr.jl"),
                            l2=[
                                0.010014531529951328,
                                0.0176268986746271,
                                0.01817514447099777,
                                0.018271085903740675,
                                0.15193033077438198
                            ],
                            linf=[
                                0.2898958869606375,
                                0.529717119064458,
                                0.5567193302705906,
                                0.570663236219957,
                                3.5496520808512027
                            ],
                            tspan=(0.0, 0.025),
                            coverage_override=(maxiters = 6,))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
        # Check for conservation
        state_integrals = Trixi.integrate(sol.u[2], semi)
        initial_state_integrals = analysis_callback.affect!.initial_state_integrals

        @test isapprox(state_integrals[1], initial_state_integrals[1], atol = 1e-13)
        @test isapprox(state_integrals[2], initial_state_integrals[2], atol = 1e-13)
        @test isapprox(state_integrals[3], initial_state_integrals[3], atol = 1e-13)
        @test isapprox(state_integrals[4], initial_state_integrals[4], atol = 1e-13)
        @test isapprox(state_integrals[5], initial_state_integrals[5], atol = 1e-13)
    end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
