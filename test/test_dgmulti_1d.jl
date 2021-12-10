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

  @trixi_testset "elixir_euler_flux_diff.jl (SBP) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
      cells_per_dimension = (16,),
      approximation_type = SBP(),
      l2 = [6.437827414849647e-6, 2.1840558851820947e-6, 1.3245669629438228e-5],
      linf = [2.0715843751295537e-5, 8.519520630301258e-6, 4.2642194098885255e-5]
    )
  end

  @trixi_testset "elixir_euler_flux_diff.jl (FD SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
      cells_per_dimension = (4,),
      approximation_type = derivative_operator(
        SummationByPartsOperators.MattssonNordström2004(),
        derivative_order=1, accuracy_order=4,
        xmin=0.0, xmax=1.0, N=16),
      l2 = [1.8684509287853788e-5, 1.0641411823379635e-5, 5.178010291876143e-5],
      linf = [6.933493585936645e-5, 3.0277366229292113e-5, 0.0002220020568932668]
    )
    show(stdout, semi.solver.basis)
    show(stdout, MIME"text/plain"(), semi.solver.basis)
  end

  @trixi_testset "elixir_euler_fdsbp_periodic.jl (FD SBP periodic)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
      l2 = [9.143049271995802e-7, 1.8989162075330386e-6, 3.9897318713851825e-6],
      linf = [1.7314023346148844e-6, 3.3238808521129926e-6, 6.522518362750418e-6]
    )
    show(stdout, semi.solver.basis)
    show(stdout, MIME"text/plain"(), semi.solver.basis)
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
