module TestElixirs

using LinearAlgebra
using Test
using Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples")


@testset "Special elixirs" begin
  @testset "Convergence test" begin
    mean_values = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_advection_extended.jl"), 3)
    @test isapprox(mean_values[:l2], [4.0], rtol=0.01)

    mean_values = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "paper-self-gravitating-gas-dynamics", "elixir_eulergravity_eoc_test.jl"), 2, tspan=(0.0, 0.1))
    @test isapprox(mean_values[:l2], 4 * ones(4), atol=0.4)
  end


  @testset "Test linear structure (2D)" begin
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_advection_extended.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=2)
    A, b = linear_structure(semi)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))

    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_hypdiff_lax_friedrichs.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=2)
    A, b = linear_structure(semi)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))

    # check whether the user can modify `b` without changing `A`
    x = vec(ode.u0)
    Ax = A * x
    @. b = 2 * b + x
    @test A * x ≈ Ax
  end


  @testset "Test Jacobian of DG (2D)" begin
    @testset "Linear advection" begin
      trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_advection_extended.jl"),
                    tspan=(0.0, 0.0), initial_refinement_level=2)
      A, _ = linear_structure(semi)

      J = jacobian_ad_forward(semi)
      @test Matrix(A) ≈ J
      λ = eigvals(J)
      @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))

      J = jacobian_fd(semi)
      @test Matrix(A) ≈ J
      λ = eigvals(J)
      @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))
    end

    @testset "Compressible Euler equations" begin
      trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_euler_density_wave.jl"),
                    tspan=(0.0, 0.0), initial_refinement_level=2)

      J = jacobian_ad_forward(semi)
      λ = eigvals(J)
      @test maximum(real, λ) < 7.0e-7

      J = jacobian_fd(semi)
      λ = eigvals(J)
      @test maximum(real, λ) < 7.0e-3


      trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_euler_shockcapturing.jl"),
                    tspan=(0.0, 0.0), initial_refinement_level=1)
      # This does not work yet because of the indicators...
      @test_skip jacobian_ad_forward(semi)
    end

    @testset "MHD" begin
      trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_mhd_alfven_wave.jl"),
                    tspan=(0.0, 0.0), initial_refinement_level=1)
      @test_nowarn jacobian_ad_forward(semi)

      trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_mhd_alfven_wave_mortar.jl"),
                    tspan=(0.0, 0.0), initial_refinement_level=1)
      @test_nowarn jacobian_ad_forward(semi)
    end
  end


  @testset "Test linear structure (3D)" begin
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "3d", "elixir_advection_extended.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=1)
    A, b = linear_structure(semi)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))
  end


  @testset "Test Jacobian of DG (3D)" begin
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "3d", "elixir_advection_extended.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=1)
    A, _ = linear_structure(semi)

    J = jacobian_ad_forward(semi)
    @test Matrix(A) ≈ J
    λ = eigvals(J)
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))

    J = jacobian_fd(semi)
    @test Matrix(A) ≈ J
    λ = eigvals(J)
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
