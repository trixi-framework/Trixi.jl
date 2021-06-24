module TestElixirs

using LinearAlgebra
using Test
using Trixi

import ForwardDiff

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples")


@testset "Special elixirs" begin
  @testset "Convergence test" begin
    mean_eoc = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_advection_extended.jl"), 3)
    @test isapprox(mean_eoc[:l2], [4.0], rtol=0.01)

    mean_eoc = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_advection_extended_curved.jl"), 3)
    @test isapprox(mean_eoc[:l2], [4.0], rtol=0.01)

    mean_eoc = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_euler_nonperiodic_p4est_unstructured.jl"), 3)
    @test isapprox(mean_eoc[:l2], [3.54, 3.50, 3.50, 3.52], rtol=0.01)

    mean_eoc = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "3d", "elixir_advection_basic_curved.jl"), 2)
    @test isapprox(mean_eoc[:l2], [4.0], rtol=0.01)

    mean_eoc = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "3d", "elixir_advection_p4est_unstructured_curved.jl"), 2, initial_refinement_level=1)
    @test isapprox(mean_eoc[:l2], [3.31], rtol=0.01)

    mean_eoc = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "paper-self-gravitating-gas-dynamics", "elixir_eulergravity_eoc_test.jl"), 2, tspan=(0.0, 0.1))
    @test isapprox(mean_eoc[:l2], 4 * ones(4), atol=0.4)
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


  @testset "AD using ForwardDiff" begin
    @testset "Euler equations 1D" begin
      function entropy_at_final_time(k) # k is the wave number of the initial condition
        equations = CompressibleEulerEquations1D(1.4)
        mesh = TreeMesh((-1.0,), (1.0,), initial_refinement_level=3, n_cells_max=10^4)
        solver = DGSEM(3, flux_hll, VolumeIntegralFluxDifferencing(flux_ranocha))
        initial_condition = (x, t, equations) -> begin
          rho = 2 + sinpi(k * sum(x))
          v1  = 0.1
          p   = 10.0
          return prim2cons(SVector(rho, v1, p), equations)
        end
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                            uEltype=typeof(k))
        ode = semidiscretize(semi, (0.0, 1.0))
        summary_callback = SummaryCallback()
        analysis_interval = 100
        analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
        alive_callback = AliveCallback(analysis_interval=analysis_interval)
        callbacks = CallbackSet(
            summary_callback,
            analysis_callback,
            alive_callback
        )
        sol = solve(ode, SSPRK43(), callback=callbacks)
        Trixi.integrate(entropy, sol.u[end], semi)
      end
      ForwardDiff.derivative(entropy_at_final_time, 1.0) ≈ -0.4524664696235628
    end

    @testset "Linear advection 2D" begin
      function energy_at_final_time(k) # k is the wave number of the initial condition
        equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
        mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4)
        solver = DGSEM(3, flux_lax_friedrichs)
        initial_condition = (x, t, equation) -> begin
          x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
          return SVector(sinpi(k * sum(x_trans)))
        end
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                            uEltype=typeof(k))
        ode = semidiscretize(semi, (0.0, 1.0))
        summary_callback = SummaryCallback()
        analysis_interval = 100
        analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
        alive_callback = AliveCallback(analysis_interval=analysis_interval)
        stepsize_callback = StepsizeCallback(cfl=1.6)
        callbacks = CallbackSet(
            summary_callback,
            analysis_callback,
            alive_callback,
            stepsize_callback
        )
        sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), save_everystep=false, adaptive=false, dt=1.0, callback=callbacks)
        Trixi.integrate(energy_total, sol.u[end], semi)
      end
      ForwardDiff.derivative(energy_at_final_time, 1.0) ≈ 1.4388628342896945e-5
    end
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
