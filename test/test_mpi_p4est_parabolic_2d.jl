module TestExamplesMPIP4estMesh2DParabolic

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

@testset "P4estMesh MPI 2D Parabolic" begin
    @trixi_testset "P4estMesh2D: elixir_navierstokes_lid_driven_cavity.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_lid_driven_cavity.jl"),
                            initial_refinement_level=2, tspan=(0.0, 0.5),
                            l2=[
                                0.00028716166408816073,
                                0.08101204560401647,
                                0.02099595625377768,
                                0.05008149754143295
                            ],
                            linf=[
                                0.014804500261322406,
                                0.9513271652357098,
                                0.7223919625994717,
                                1.4846907331004786
                            ])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh2D: elixir_navierstokes_convergence_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_convergence_nonperiodic.jl"),
                            initial_refinement_level=1, tspan=(0.0, 0.2),
                            l2=[
                                0.0004036496258545996,
                                0.0005869762480189079,
                                0.0009148853742181908,
                                0.0011984191532764543
                            ],
                            linf=[
                                0.0024993634989209923,
                                0.009487866203496731,
                                0.004505829506103787,
                                0.011634902753554499
                            ])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh2D: elixir_advection_diffusion_nonperiodic_curved.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_diffusion_nonperiodic_curved.jl"),
                            trees_per_dimension=(1, 1), initial_refinement_level=2,
                            tspan=(0.0, 0.5),
                            l2=[0.00919917034843865],
                            linf=[0.14186297438393505])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_diffusion_periodic.jl"),
                            trees_per_dimension=(1, 1), initial_refinement_level=2,
                            tspan=(0.0, 0.5),
                            l2=[0.0023754695605828443],
                            linf=[0.008154128363741964])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "elixir_navierstokes_NACA0012airfoil_mach08.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                            l2=[0.000186486564226516,
                                0.0005076712323400374,
                                0.00038074588984354107,
                                0.002128177239782089],
                            linf=[0.5153387072802718,
                                1.199362305026636,
                                0.9077214424040279,
                                5.666071182328691], tspan=(0.0, 0.001),
                            initial_refinement_level=0)

        u_ode = copy(sol.u[end])
        du_ode = zero(u_ode) # Just a placeholder in this case

        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)

        drag_p = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                               semi.cache, semi)
        lift_p = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                               semi.cache, semi)

        drag_f = Trixi.analyze(drag_coefficient_shear_force, du, u, tspan[2], mesh,
                               equations, equations_parabolic, solver,
                               semi.cache, semi, semi.cache_parabolic)
        lift_f = Trixi.analyze(lift_coefficient_shear_force, du, u, tspan[2], mesh,
                               equations, equations_parabolic, solver,
                               semi.cache, semi, semi.cache_parabolic)

        @test isapprox(drag_p, 0.17963843913309516, atol = 1e-13)
        @test isapprox(lift_p, 0.26462588007949367, atol = 1e-13)

        @test isapprox(drag_f, 1.5427441885921553, atol = 1e-13)
        @test isapprox(lift_f, 0.005621910087395724, atol = 1e-13)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        # We move these tests here to avoid modifying values used
        # to compute the drag/lift coefficients above.
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end
end
end # module
