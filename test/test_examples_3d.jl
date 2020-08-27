module TestExamples3D

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
  @testset "parameters_ec.toml with flux_chandrashekar" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec.toml"),
            l2   = [0.025105743648126774, 0.016571417754430256, 0.01657141775443023, 0.016565202090289916, 0.09077232065771225],
            linf = [0.4349225166034201, 0.27945714200874, 0.2794571420087401, 0.28021366413271664, 1.5240679700745954],
            surface_flux=flux_chandrashekar, volume_flux=flux_chandrashekar)
  end
  @testset "parameters_ec.toml with flux_kennedry_gruber" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec.toml"),
            l2   = [0.025120431810845507, 0.016599310737401483, 0.01659931073740148, 0.016592567464138185, 0.090856457771812],
            linf = [0.43120500632996794, 0.28419288751363336, 0.2841928875136334, 0.28583515705222146, 1.515485025725378],
            surface_flux=flux_kennedy_gruber, volume_flux=flux_kennedy_gruber)
  end
  @testset "parameters_ec.toml with flux_shima_etal" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec.toml"),
            l2   = [0.025158572247744777, 0.016638273163668674, 0.016638273163668692, 0.016631581587765128, 0.09097798883433982],
            linf = [0.4298582725942134, 0.28561096934734514, 0.28561096934734487, 0.287502111001965, 1.5313368512936554],
            surface_flux=flux_shima_etal, volume_flux=flux_shima_etal)
  end
  @testset "parameters_taylor_green_vortex.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_taylor_green_vortex.toml"),
            l2   = [0.0003494971047256544, 0.03133386380969968, 0.031333863809699644, 0.04378595081016185, 0.015796569210801217],
            linf = [0.0013934701399120897, 0.07284947983025436, 0.07284947983025408, 0.12803234075782724, 0.07624639122292365],
            t_end = 0.5)
  end
  @testset "parameters_shock_capturing.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_shock_capturing.toml"),
            l2   = [0.025558219399128387, 0.01612806446620796, 0.016128064466207948, 0.016120400619198158, 0.09208276987000782],
            linf = [0.3950327737713353, 0.26324766244272796, 0.2632476624427279, 0.2634129727753079, 1.371321006006725])
  end
  @testset "parameters_hyp_diff_llf.toml with initial_refinement_level=2" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_llf.toml"),
            l2   = [0.0015303292770225546, 0.011314166522881952, 0.011314166522881981, 0.011314166522881947],
            linf = [0.022634590339093097, 0.10150613595329361, 0.10150613595329361, 0.10150613595329361],
            initial_refinement_level=2)
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
