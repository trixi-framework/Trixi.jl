module TestExamplesDGMulti1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_1d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 1D" begin

  @trixi_testset "elixir_advection_gauss_sbp.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gauss_sbp.jl"),
      cells_per_dimension = (8,),
      l2 = [2.9953644500009865e-5],
      linf = [4.467840577382365e-5]
    )
  end

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

  @trixi_testset "elixir_euler_fdsbp_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
      l2 = [9.146929180585711e-7, 1.8997616878017292e-6, 3.991417702211889e-6],
      linf = [1.7321089884614338e-6, 3.3252888855805907e-6, 6.5252787737613005e-6]
    )
    show(stdout, semi.solver.basis)
    show(stdout, MIME"text/plain"(), semi.solver.basis)
  end

  @trixi_testset "DGMulti with periodic SBP unit test" begin
    # see https://github.com/trixi-framework/Trixi.jl/pull/1013
    dg = DGMulti(element_type = Line(),
                 approximation_type = periodic_derivative_operator(
                   derivative_order=1, accuracy_order=4, xmin=-5.0, xmax=10.0, N=50))
    mesh = DGMultiMesh(dg)
    @test mapreduce(isapprox, &, mesh.md.xyz, dg.basis.rst)
    # check to make sure nodes are rescaled to [-1, 1]
    @test minimum(dg.basis.rst[1]) ≈ -1
    @test maximum(dg.basis.rst[1]) ≈ 1 atol=0.35
  end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
