using Test
import Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)

# Run basic tests
@testset "Examples (short execution time)" begin
  @testset "../examples/euler_gravity_paper/parameters_no_gravity_manufac.toml" begin
    test_trixi_run("../examples/euler_gravity_paper/parameters_no_gravity_manufac.toml",
           l2   = [1.7203325428717625e-5, 2.333262549313786e-5, 2.3332625493309535e-5, 4.37888222568518e-5],
           linf = [0.00011871318930278818, 0.0001630056973325189, 0.00016300569733962433, 0.00036579406041914453])
  end
  @testset "../examples/euler_gravity_paper/parameters_no_gravity_manufac.toml with N=4" begin
    test_trixi_run("../examples/euler_gravity_paper/parameters_no_gravity_manufac.toml",
           l2   = [6.819409408806212e-7, 8.920817115518294e-7, 8.920817113925391e-7, 1.6782615286312978e-6],
           linf = [5.8192832639303305e-6, 7.2422583947684416e-6, 7.242258397877066e-6, 1.5397746985446048e-5],
           N=4)
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)
