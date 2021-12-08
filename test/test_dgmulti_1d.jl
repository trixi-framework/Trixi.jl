module TestExamplesDGMulti1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_1d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 1D" begin
  @trixi_testset "elixir_euler_flux_diff.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
      cells_per_dimension = (16,),
      # division by sqrt(2.0) corresponds to normalization by the square root of the size of the domain
      l2 = [7.853842541289665e-7, 9.609905503440606e-7, 2.832322219966481e-6] ./ sqrt(2.0),
      linf = [1.5003758788711963e-6, 1.802998748523521e-6, 4.83599270806323e-6]
    )
  end

  @trixi_testset "elixir_euler_flux_diff.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"), 3)
    @test isapprox(mean_convergence[:l2], [4.1558759698638434, 3.977911306037128, 4.041421206468769], rtol=0.05)
  end

  @trixi_testset "elixir_euler_flux_diff.jl (FD SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
      cells_per_dimension = (4,),
      approximation_type = derivative_operator(
        SummationByPartsOperators.MattssonNordström2004(),
        derivative_order=1, accuracy_order=4,
        xmin=0.0, xmax=1.0, N=16),
      l2 = [2.6417669774800198e-5, 1.5028052071001864e-5, 7.321424811919818e-5] ./ sqrt(2.0),
      linf = [6.923356213395238e-5, 3.008425995654207e-5, 0.0002216469993250314]
    )
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
