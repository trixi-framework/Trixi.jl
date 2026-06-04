module TestExamplesMPIP4estMesh3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

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
            mpi_isroot() && println("-"^100)
            mpi_isroot() &&
                println("elixir_advection_basic.jl with error-based step size control")

            # Use callbacks without stepsize_callback to test error-based step size control
            callbacks = CallbackSet(summary_callback, analysis_callback, save_restart,
                                    save_solution)
            sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                        ode_default_options()..., callback = callbacks)
            summary_callback()
            errors = analysis_callback(sol)
            if mpi_isroot()
                @test errors.l2≈[0.00016800412839949264] rtol=1.0e-4
                @test errors.linf≈[0.0014548839020096516] rtol=1.0e-4
            end
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
    end

    @trixi_testset "elixir_advection_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[9.773852895157622e-6],
                            linf=[0.0005853874124926162])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
    end

    @trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_unstructured_curved.jl"),
                            l2=[1.6163120948209677e-5],
                            linf=[0.0010572201890564834],
                            tspan=(0.0, 1.0))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
    end

    @trixi_testset "elixir_advection_restart.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                            l2=[0.002590388934758452],
                            linf=[0.01840757696885409])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
    end

    @trixi_testset "elixir_advection_cubed_sphere.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
                            l2=[0.002006918015656413],
                            linf=[0.027655117058380085])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
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
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
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
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
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
                            tspan=(0.0, 0.2))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
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
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
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
                            tspan=(0.0, 0.25))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
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
                        tspan=(0.0, 0.25), trees_per_dimension=(1, 1, 1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "MPI 3D, combine_conservative_and_nonconservative_fluxes" begin
    @muladd @inline function flux_hindenlang_gassner_nonconservative_powell(u_ll, u_rr,
                                                                            normal_direction::AbstractVector,
                                                                            equations::IdealGlmMhdEquations3D)
        # Unpack left and right states
        rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll,
                                                                                   equations)
        rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr,
                                                                                   equations)
        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                     v3_ll * normal_direction[3]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                     v3_rr * normal_direction[3]
        B_dot_n_ll = B1_ll * normal_direction[1] + B2_ll * normal_direction[2] +
                     B3_ll * normal_direction[3]
        B_dot_n_rr = B1_rr * normal_direction[1] + B2_rr * normal_direction[2] +
                     B3_rr * normal_direction[3]

        # Compute the necessary mean values needed for either direction
        rho_mean = Trixi.ln_mean(rho_ll, rho_rr)
        # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
        # in exact arithmetic since
        #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
        #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
        inv_rho_p_mean = p_ll * p_rr * Trixi.inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
        v1_avg = 0.5f0 * (v1_ll + v1_rr)
        v2_avg = 0.5f0 * (v2_ll + v2_rr)
        v3_avg = 0.5f0 * (v3_ll + v3_rr)
        p_avg = 0.5f0 * (p_ll + p_rr)
        psi_avg = 0.5f0 * (psi_ll + psi_rr)
        velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
        magnetic_square_avg = 0.5f0 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

        # Calculate fluxes depending on normal_direction
        f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
        f2 = (f1 * v1_avg + (p_avg + magnetic_square_avg) * normal_direction[1]
              -
              0.5f0 * (B_dot_n_ll * B1_rr + B_dot_n_rr * B1_ll))
        f3 = (f1 * v2_avg + (p_avg + magnetic_square_avg) * normal_direction[2]
              -
              0.5f0 * (B_dot_n_ll * B2_rr + B_dot_n_rr * B2_ll))
        f4 = (f1 * v3_avg + (p_avg + magnetic_square_avg) * normal_direction[3]
              -
              0.5f0 * (B_dot_n_ll * B3_rr + B_dot_n_rr * B3_ll))
        #f5 below
        f6 = (equations.c_h * psi_avg * normal_direction[1]
              +
              0.5f0 * (v_dot_n_ll * B1_ll - v1_ll * B_dot_n_ll +
               v_dot_n_rr * B1_rr - v1_rr * B_dot_n_rr))
        f7 = (equations.c_h * psi_avg * normal_direction[2]
              +
              0.5f0 * (v_dot_n_ll * B2_ll - v2_ll * B_dot_n_ll +
               v_dot_n_rr * B2_rr - v2_rr * B_dot_n_rr))
        f8 = (equations.c_h * psi_avg * normal_direction[3]
              +
              0.5f0 * (v_dot_n_ll * B3_ll - v3_ll * B_dot_n_ll +
               v_dot_n_rr * B3_rr - v3_rr * B_dot_n_rr))
        f9 = equations.c_h * 0.5f0 * (B_dot_n_ll + B_dot_n_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (f1 *
              (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
              +
              0.5f0 * (+p_ll * v_dot_n_rr + p_rr * v_dot_n_ll
               + (v_dot_n_ll * B1_ll * B1_rr + v_dot_n_rr * B1_rr * B1_ll)
               + (v_dot_n_ll * B2_ll * B2_rr + v_dot_n_rr * B2_rr * B2_ll)
               + (v_dot_n_ll * B3_ll * B3_rr + v_dot_n_rr * B3_rr * B3_ll)
               -
               (v1_ll * B_dot_n_ll * B1_rr + v1_rr * B_dot_n_rr * B1_ll)
               -
               (v2_ll * B_dot_n_ll * B2_rr + v2_rr * B_dot_n_rr * B2_ll)
               -
               (v3_ll * B_dot_n_ll * B3_rr + v3_rr * B_dot_n_rr * B3_ll)
               +
               equations.c_h * (B_dot_n_ll * psi_rr + B_dot_n_rr * psi_ll)))

        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr
        f = SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
        g_left = SVector(0,
                         B1_ll * B_dot_n_rr,
                         B2_ll * B_dot_n_rr,
                         B3_ll * B_dot_n_rr,
                         v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
                         v1_ll * B_dot_n_rr,
                         v2_ll * B_dot_n_rr,
                         v3_ll * B_dot_n_rr,
                         v_dot_n_ll * psi_rr)

        g_right = SVector(0,
                          B1_rr * B_dot_n_ll,
                          B2_rr * B_dot_n_ll,
                          B3_rr * B_dot_n_ll,
                          v_dot_B_rr * B_dot_n_ll + v_dot_n_rr * psi_rr * psi_ll,
                          v1_rr * B_dot_n_ll,
                          v2_rr * B_dot_n_ll,
                          v3_rr * B_dot_n_ll,
                          v_dot_n_rr * psi_ll)
        flux_left = f + 0.5f0 * g_left
        flux_right = f + 0.5f0 * g_right
        return flux_left, flux_right
    end

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_hindenlang_gassner_nonconservative_powell),
    equations::IdealGlmMhdEquations3D) = Trixi.True()

    @muladd @inline function flux_hlle_nonconservative_powell(u_ll, u_rr,
                                                              normal_direction::AbstractVector,
                                                              equations::IdealGlmMhdEquations3D)
        f = flux_hlle(u_ll, u_rr, normal_direction, equations)

        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_total_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_total_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

        v1_ll = rho_v1_ll / rho_ll
        v2_ll = rho_v2_ll / rho_ll
        v3_ll = rho_v3_ll / rho_ll
        v1_rr = rho_v1_rr / rho_rr
        v2_rr = rho_v2_rr / rho_rr
        v3_rr = rho_v3_rr / rho_rr
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                     v3_ll * normal_direction[3]
        B_dot_n_rr = B1_rr * normal_direction[1] +
                     B2_rr * normal_direction[2] +
                     B3_rr * normal_direction[3]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                     v3_rr * normal_direction[3]
        B_dot_n_ll = B1_ll * normal_direction[1] +
                     B2_ll * normal_direction[2] +
                     B3_ll * normal_direction[3]

        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
        g_left = SVector(0,
                         B1_ll * B_dot_n_rr,
                         B2_ll * B_dot_n_rr,
                         B3_ll * B_dot_n_rr,
                         v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
                         v1_ll * B_dot_n_rr,
                         v2_ll * B_dot_n_rr,
                         v3_ll * B_dot_n_rr,
                         v_dot_n_ll * psi_rr)

        g_right = SVector(0,
                          B1_rr * B_dot_n_ll,
                          B2_rr * B_dot_n_ll,
                          B3_rr * B_dot_n_ll,
                          v_dot_B_rr * B_dot_n_ll + v_dot_n_rr * psi_rr * psi_ll,
                          v1_rr * B_dot_n_ll,
                          v2_rr * B_dot_n_ll,
                          v3_rr * B_dot_n_ll,
                          v_dot_n_rr * psi_ll)
        flux_left = f + 0.5f0 * g_left
        flux_right = f + 0.5f0 * g_right

        return flux_left, flux_right
    end

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_hlle_nonconservative_powell),
    equations::IdealGlmMhdEquations3D) = Trixi.True()

    @inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner,
                                                                      normal_direction::AbstractVector,
                                                                      x, t,
                                                                      surface_flux_function::typeof(flux_hlle_nonconservative_powell),
                                                                      equations)

        # get the external value of the solution
        u_boundary = boundary_condition.boundary_value_function(x, t, equations)

        # Calculate boundary flux
        flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction,
                                        equations)
        return flux
    end

    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonperiodic.jl"),
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
                        surface_flux=flux_hlle_nonconservative_powell,
                        volume_integral=VolumeIntegralFluxDifferencing(flux_hindenlang_gassner_nonconservative_powell),
                        tspan=(0.0, 0.1))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # P4estMesh MPI

end # module
