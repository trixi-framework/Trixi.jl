module TestExamplesParallel2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
Trixi.is_mpi_root() && isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

# Run basic tests
@testset "Examples 2D" begin
  @testset "parameters.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [9.144681765639205e-6],
            linf = [6.437440532547356e-5])
  end
  @testset "parameters.toml with polydeg=1" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [0.05264106093598111],
            linf = [0.08754218386076518],
            polydeg=1)
  end
  @testset "parameters_ec.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec.toml"),
            l2   = [0.06159341742582756, 0.05012484425381723, 0.05013298724507752, 0.22537740506116724],
            linf = [0.29912627861573327, 0.30886767304359375, 0.3088108573487326, 1.0657556075017878])
  end
  @testset "parameters_density_wave.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_density_wave.toml"),
            l2   = [0.001060077845747576, 0.00010600778457107525, 0.00021201556914875742, 2.6501946139091318e-5],
            linf = [0.0065356386867677085, 0.0006535638688170142, 0.0013071277374487877, 0.0001633909674296774],
            extra_analysis_quantities=["l2_error_primitive", "linf_error_primitive"], t_end=0.5)
  end
  @testset "parameters_ec_mhd.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec_mhd.toml"),
            l2   = [0.03607862694368351, 0.04281395008247395, 0.04280207686965749, 0.025746770192645763, 0.1611518499414067, 0.017455917249117023, 0.017456981264942977, 0.02688321120361229, 0.00015024027267648003],
            linf = [0.23502083666166018, 0.3156846367743936, 0.31227895161037256, 0.2118146956106238, 0.9743049414302711, 0.09050624115026618, 0.09131633488909774, 0.15693063355520998, 0.0038394720095667593])
  end
  @testset "parameters_hyp_diff_harmonic_nonperiodic.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_harmonic_nonperiodic.toml"),
            l2   = [8.618132353932638e-8, 5.619399844708813e-7, 5.619399845476024e-7],
            linf = [1.124861862326869e-6, 8.622436471483752e-6, 8.622436469707395e-6])
  end
  @testset "parameters_hyp_diff_llf.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_llf.toml"),
            l2   = [0.00015687751088073104, 0.0010259867353397119, 0.0010259867353398994],
            linf = [0.001198695640053704, 0.006423873515701395, 0.006423873515686296])
  end
  @testset "parameters_hyp_diff_nonperiodic.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_nonperiodic.toml"),
            l2   = [8.523077654037775e-6, 2.877932365308637e-5, 5.454942769137812e-5],
            linf = [5.484978959957587e-5, 0.00014544895979200218, 0.000324491268921534])
  end
  @testset "parameters_hyp_diff_upwind.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_upwind.toml"),
            l2   = [5.868147556488962e-6, 3.8051792732628014e-5, 3.8051792732620214e-5],
            linf = [3.70196549871471e-5, 0.0002072058411455302, 0.00020720584114464202])
  end
  @testset "parameters_nonperiodic.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_nonperiodic.toml"),
            l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
            linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5])
  end
  @testset "parameters_source_terms.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_source_terms.toml"),
            l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
            linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5])
  end
end

# Clean up afterwards: delete Trixi output directory
Trixi.is_mpi_root() && @test_nowarn rm(outdir, recursive=true)

end #module
