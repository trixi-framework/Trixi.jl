module TestExamples

using Test
import Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

# Run basic tests
@testset "Examples 3D" begin
  @testset "parameters.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [0.00015975754755823664],
            linf = [0.001503873297666436])
  end
  @testset "parameters_source_terms.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_source_terms.toml"),
            l2   = [0.010323099666828388, 0.00972876713766357, 0.00972876713766343, 0.009728767137663324, 0.015080409341036285],
            linf = [0.034894880154510144, 0.03383545920056008, 0.033835459200560525, 0.03383545920054587, 0.06785780622711979])
  end
  @testset "parameters_source_terms.toml with split_form" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_source_terms.toml"),
            l2   = [0.010323099666828388, 0.00972876713766357, 0.00972876713766343, 0.009728767137663324, 0.015080409341036285],
            linf = [0.034894880154510144, 0.03383545920056008, 0.033835459200560525, 0.03383545920054587, 0.06785780622711979],
            volume_integral_type = "split_form")
  end
  @testset "parameters_mortar.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mortar.toml"),
            l2   = [0.0018461483161353273],
            linf = [0.017728496545256434])
  end
  @testset "parameters_mortar_euler.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mortar_euler.toml"),
            l2   = [0.0019011097431965655, 0.0018289464087588392, 0.0018289464087585998, 0.0018289464087588862, 0.003354766311541738],
            linf = [0.011918594206950184, 0.011808582644224241, 0.011808582644249999, 0.011808582644239785, 0.02464803617735356])
  end
  @testset "parameters_amr.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_amr.toml"),
            l2   = [9.773858425669403e-6],
            linf = [0.0005853874124926092])
  end
  @testset "parameters_ec.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec.toml"),
            l2   = [0.025101741317688664, 0.01655620530022176, 0.016556205300221737, 0.016549388264402515, 0.09075092792976944],
            linf = [0.43498932208478724, 0.2821813924028202, 0.28218139240282025, 0.2838043627560838, 1.5002293438086647])
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
