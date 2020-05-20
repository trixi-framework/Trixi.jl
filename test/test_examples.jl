using Test
import Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)

# Run basic tests
@testset "Examples (short execution time)" begin
  # when #41 is merged, we can test final errors here as @test final_error ≈ expected_error
  @test_nowarn Trixi.run("../examples/parameters.toml")
  @test_nowarn Trixi.run("../examples/parameters_alfven_wave.toml")
  @test_nowarn Trixi.run("../examples/parameters_amr.toml")
  @test_nowarn Trixi.run("../examples/parameters_amr_nonperiodic.toml")
  @test_nowarn Trixi.run("../examples/parameters_amr_vortex.toml")
  @test_nowarn Trixi.run("../examples/parameters_blast_wave_shockcapturing.toml")
  @test_skip   Trixi.run("../examples/parameters_blast_wave_shockcapturing_amr.toml") # errors for me
  @test_nowarn Trixi.run("../examples/parameters_ec.toml")
  @test_nowarn Trixi.run("../examples/parameters_ec_mhd.toml")
  @test_nowarn Trixi.run("../examples/parameters_hyp_diff_harmonic_nonperiodic.toml")
  @test_nowarn Trixi.run("../examples/parameters_hyp_diff_llf.toml")
  @test_nowarn Trixi.run("../examples/parameters_hyp_diff_nonperiodic.toml")
  @test_nowarn Trixi.run("../examples/parameters_hyp_diff_upwind.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar_vortex.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar_vortex_split.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar_vortex_split_shockcapturing.toml")
  @test_nowarn Trixi.run("../examples/parameters_nonperiodic.toml")
  @test_skip   Trixi.run("../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml") # errors for me
  @test_nowarn Trixi.run("../examples/parameters_source_terms.toml")
  @test_nowarn Trixi.run("../examples/parameters_vortex.toml")
  @test_nowarn Trixi.run("../examples/parameters_vortex_split_shockcapturing.toml")
  @test_nowarn Trixi.run("../examples/parameters_weak_blast_wave_shockcapturing.toml")
  @test_nowarn Trixi.convtest("../examples/parameters.toml", 3)
end

# Only run extended tests if environment variable is set
if haskey(ENV, "TRIXI_TEST_EXTENDED") && lowercase(ENV["TRIXI_TEST_EXTENDED"]) in ("1", "on", "yes")
  @testset "Examples (long execution time)" begin
    @test_nowarn Trixi.run("../examples/parameters_blob.toml")
    @test_nowarn Trixi.run("../examples/parameters_blob_amr.toml")
    @test_nowarn Trixi.run("../examples/parameters_ec_performance_test.toml")
    @test_nowarn Trixi.run("../examples/parameters_khi.toml")
    @test_nowarn Trixi.run("../examples/parameters_ec_mortar.toml")
    @test_nowarn Trixi.run("../examples/parameters_khi_amr.toml")
    @test_nowarn Trixi.run("../examples/parameters_mhd_blast_wave.toml")
    @test_nowarn Trixi.run("../examples/parameters_orszag_tang.toml")
    @test_nowarn Trixi.run("../examples/parameters_rotor.toml")
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)
