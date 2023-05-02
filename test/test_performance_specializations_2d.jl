module TestPerformanceSpecializations2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)


@testset "Performance specializations 2D" begin
  @timed_testset "TreeMesh2D, flux_shima_etal_turbo" begin
    trixi_include(@__MODULE__,
      joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"),
      initial_refinement_level=0, tspan=(0.0, 0.0), polydeg=3,
      volume_flux=flux_shima_etal_turbo, surface_flux=flux_shima_etal_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
      u = Trixi.wrap_array(u_ode, semi)
      du = Trixi.wrap_array(du_ode, semi)
      nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

      # Call the optimized default version
      du .= 0
      Trixi.flux_differencing_kernel!(
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_specialized = du[:, :, :, 1]

      # Call the plain version - note the argument type `Function` of
      # `semi.solver.volume_integral.volume_flux`
      du .= 0
      invoke(Trixi.flux_differencing_kernel!,
        Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
        typeof(nonconservative_terms), typeof(semi.equations),
        Function, typeof(semi.solver), typeof(semi.cache), Bool},
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_baseline = du[:, :, :, 1]

      @test du_specialized ≈ du_baseline
    end
  end

  @timed_testset "TreeMesh2D, flux_ranocha_turbo" begin
    trixi_include(@__MODULE__,
      joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"),
      initial_refinement_level=0, tspan=(0.0, 0.0), polydeg=3,
      volume_flux=flux_ranocha_turbo, surface_flux=flux_ranocha_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
      u = Trixi.wrap_array(u_ode, semi)
      du = Trixi.wrap_array(du_ode, semi)
      nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

      # Call the optimized default version
      du .= 0
      Trixi.flux_differencing_kernel!(
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_specialized = du[:, :, :, 1]

      # Call the plain version - note the argument type `Function` of
      # `semi.solver.volume_integral.volume_flux`
      du .= 0
      invoke(Trixi.flux_differencing_kernel!,
        Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
        typeof(nonconservative_terms), typeof(semi.equations),
        Function, typeof(semi.solver), typeof(semi.cache), Bool},
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_baseline = du[:, :, :, 1]

      @test du_specialized ≈ du_baseline
    end
  end

  @timed_testset "StructuredMesh2D, flux_shima_etal_turbo" begin
    trixi_include(@__MODULE__,
      joinpath(examples_dir(), "structured_2d_dgsem", "elixir_euler_ec.jl"),
      cells_per_dimension=(1, 1), tspan=(0.0, 0.0), polydeg=3,
      volume_flux=flux_shima_etal_turbo, surface_flux=flux_shima_etal_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
      u = Trixi.wrap_array(u_ode, semi)
      du = Trixi.wrap_array(du_ode, semi)
      nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

      # Call the optimized default version
      du .= 0
      Trixi.flux_differencing_kernel!(
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_specialized = du[:, :, :, 1]

      # Call the plain version - note the argument type `Function` of
      # `semi.solver.volume_integral.volume_flux`
      du .= 0
      invoke(Trixi.flux_differencing_kernel!,
        Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
        typeof(nonconservative_terms), typeof(semi.equations),
        Function, typeof(semi.solver), typeof(semi.cache), Bool},
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_baseline = du[:, :, :, 1]

      @test du_specialized ≈ du_baseline
    end
  end

  @timed_testset "StructuredMesh2D, flux_ranocha_turbo" begin
    trixi_include(@__MODULE__,
      joinpath(examples_dir(), "structured_2d_dgsem", "elixir_euler_ec.jl"),
      cells_per_dimension=(1, 1), tspan=(0.0, 0.0), polydeg=3,
      volume_flux=flux_ranocha_turbo, surface_flux=flux_ranocha_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
      u = Trixi.wrap_array(u_ode, semi)
      du = Trixi.wrap_array(du_ode, semi)
      nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

      # Call the optimized default version
      du .= 0
      Trixi.flux_differencing_kernel!(
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_specialized = du[:, :, :, 1]

      # Call the plain version - note the argument type `Function` of
      # `semi.solver.volume_integral.volume_flux`
      du .= 0
      invoke(Trixi.flux_differencing_kernel!,
        Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
        typeof(nonconservative_terms), typeof(semi.equations),
        Function, typeof(semi.solver), typeof(semi.cache), Bool},
        du, u, 1, semi.mesh,
        nonconservative_terms, semi.equations,
        semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
      du_baseline = du[:, :, :, 1]

      @test du_specialized ≈ du_baseline
    end
  end
end


# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive=true)

end #module
