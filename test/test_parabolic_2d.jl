module TestExamplesParabolic2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "SemidiscretizationHyperbolicParabolic" begin

  @trixi_testset "DGMulti 2D rhs_parabolic!" begin

    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    mesh = DGMultiMesh(dg, cells_per_dimension=(2, 2))

    # test with polynomial initial condition x^2 * y
    # test if we recover the exact second derivative
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0, equations)

    semi = SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic, initial_condition, dg)
    @test_nowarn_debug show(stdout, semi)
    @test_nowarn_debug show(stdout, MIME"text/plain"(), semi)
    @test_nowarn_debug show(stdout, boundary_condition_do_nothing)

    @test nvariables(semi)==nvariables(equations)
    @test Base.ndims(semi)==Base.ndims(mesh)
    @test Base.real(semi)==Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # test "do nothing" BC just returns first argument
    @test boundary_condition_do_nothing(u0, nothing) == u0

    @unpack cache, cache_parabolic, equations_parabolic = semi
    @unpack u_grad = cache_parabolic
    for dim in eachindex(u_grad)
      fill!(u_grad[dim], zero(eltype(u_grad[dim])))
    end

    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    Trixi.calc_gradient!(u_grad, ode.u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, cache, cache_parabolic)
    @unpack x, y = mesh.md
    @test getindex.(u_grad[1], 1) ≈ 2 * x .* y
    @test getindex.(u_grad[2], 1) ≈ x.^2

    u_flux = similar.(u_grad)
    Trixi.calc_viscous_fluxes!(u_flux, ode.u0, u_grad, mesh, equations_parabolic,
                               dg, cache, cache_parabolic)
    @test u_flux[1] ≈ u_grad[1]
    @test u_flux[2] ≈ u_grad[2]

    du = similar(ode.u0)
    Trixi.calc_divergence!(du, ode.u0, t, u_flux, mesh, equations_parabolic, boundary_condition_periodic,
                           dg, semi.solver_parabolic, cache, cache_parabolic)
    @test getindex.(du, 1) ≈ 2 * y
  end

  @trixi_testset "DGMulti: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_advection_diffusion_periodic.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.03180371984888462],
      linf = [0.2136821621370909]
    )
  end

  @trixi_testset "DGMulti: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_advection_diffusion_nonperiodic.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.012561479036088107],
      linf = [0.10067620068384618]
    )
  end

  @trixi_testset "DGMulti: elixir_navier_stokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_navier_stokes_convergence.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.00153550768125133, 0.0033843168272696357, 0.0036531858107444067, 0.009948436427519428],
      linf = [0.005522560467190019, 0.013425258500731063, 0.013962115643483375, 0.027483102120516634]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_diffusion_nonperiodic.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.1),
      l2 = [0.0076468006081234705],
      linf = [0.10067621027072088]
    )
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
