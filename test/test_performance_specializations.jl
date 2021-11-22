module TestPerformanceSpecializations

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)


@testset "Performance specializations" begin
  @timed_testset "TreeMesh3D, flux_shima_etal" begin
    trixi_include(@__MODULE__,
      joinpath(examples_dir(), "tree_3d_dgsem", "elixir_euler_ec.jl"),
      initial_refinement_level=1, tspan=(0.0, 0.0),
      volume_flux=flux_shima_etal, surface_flux=flux_shima_etal)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)
    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)
    nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

    du .= 0
    Trixi.split_form_kernel!(
      du, u, 1, semi.mesh,
      nonconservative_terms, semi.equations,
      semi.solver.volume_integral.volume_flux, semi.solver, semi.cache)
    du_specialized = du[:, :, :, :, 1]

    du .= 0
    invoke(Trixi.split_form_kernel!,
      Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
      typeof(nonconservative_terms), typeof(semi.equations),
      Function, typeof(semi.solver), typeof(semi.cache)},
      du, u, 1, semi.mesh,
      nonconservative_terms, semi.equations,
      semi.solver.volume_integral.volume_flux, semi.solver, semi.cache)
    du_baseline = du[:, :, :, :, 1]

    @test du_specialized ≈ du_baseline
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
