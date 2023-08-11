module TestExamplesParabolic1D

using Test
using Trixi

include("test_trixi.jl")


# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "SemidiscretizationHyperbolicParabolic (1D)" begin

  @trixi_testset "TreeMesh1D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_advection_diffusion.jl"),
      initial_refinement_level = 4, tspan=(0.0, 0.4), polydeg=3,
      l2 = [8.389498188525518e-06],
      linf = [2.847421658558336e-05]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_periodic.jl"),
      l2 = [0.0001133835907077494, 6.226282245610444e-5, 0.0002820171699999139],
      linf = [0.0006255102377159538, 0.00036195501456059986, 0.0016147729485886941]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_periodic.jl"),
      equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu=mu(),
                                                                Prandtl=prandtl_number(), 
                                                                gradient_variables = GradientVariablesEntropy()),
      l2 = [0.00011310615871043463, 6.216495207074201e-5, 0.00028195843110817814],
      linf = [0.0006240837363233886, 0.0003616694320713876, 0.0016147339542413874]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_walls.jl"),
      l2 = [0.00047023310868269237, 0.00032181736027057234, 0.0014966266486095025],
      linf = [0.002996375101363302, 0.002863904256059634, 0.012691132946258676]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_walls.jl"),
      equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu=mu(),
                                                                Prandtl=prandtl_number(), 
                                                                gradient_variables = GradientVariablesEntropy()),
      l2 = [0.0004608500483647771, 0.00032431091222851285, 0.0015159733360626845],
      linf = [0.002754803146635787, 0.0028567714697580906, 0.012941794048176192]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_walls_amr.jl"),
      equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu=mu(),
                                                                Prandtl=prandtl_number()),
      l2 = [2.527877257772131e-5, 2.5539911566937718e-5, 0.0001211860451244785],
      linf = [0.00014663867588948776, 0.00019422448348348196, 0.0009556439394007299]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_walls_amr.jl"),
      equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu=mu(),
                                                                Prandtl=prandtl_number(), 
                                                                gradient_variables = GradientVariablesEntropy()),
      l2 = [2.4593699163175966e-5, 2.392863645712634e-5, 0.00011252526651714956],
      linf = [0.00011850555445525046, 0.0001898777490968537, 0.0009597561467877824]
    )
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
