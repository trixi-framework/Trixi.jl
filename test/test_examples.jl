using Test
import Trixi

if false
include("test_trixi.jl")
end

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)

# Run basic tests
@testset "Examples (short execution time)" begin
  if false
  @testset "../examples/parameters.toml" begin
    test_trixi_run("../examples/parameters.toml",
                   l2   = 2.32847750e-03,
                   linf = 4.99201998e-02)
  end
  @testset "../examples/parameters_alfven_wave.toml" begin
    test_trixi_run("../examples/parameters_alfven_wave.toml",
                   l2   = [1.11345135e-04, 5.88018891e-06, 5.88018891e-06, 8.43288100e-06, 1.29423873e-06, 1.22388203e-06, 1.22388203e-06, 1.83062175e-06, 8.08699679e-07],
                   linf = [2.56327902e-04, 1.63790212e-05, 1.63790212e-05, 2.58759953e-05, 5.32773229e-06, 8.11852027e-06, 8.11852027e-06, 1.21073548e-05, 4.17370701e-06])
  end
  end
  @test_nowarn Trixi.run("../examples/parameters.toml")
  @test_nowarn Trixi.run("../examples/parameters_alfven_wave.toml")
  @test_nowarn Trixi.run("../examples/parameters_amr.toml")
  @test_nowarn Trixi.run("../examples/parameters_amr_vortex.toml")
  @test_nowarn Trixi.run("../examples/parameters_blast_wave_shockcapturing.toml")
  @test_skip   Trixi.run("../examples/parameters_blast_wave_shockcapturing_amr.toml") # errors for me
  @test_nowarn Trixi.run("../examples/parameters_ec.toml")
  @test_nowarn Trixi.run("../examples/parameters_ec_mhd.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar_vortex.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar_vortex_split.toml")
  @test_nowarn Trixi.run("../examples/parameters_mortar_vortex_split_shockcapturing.toml")
  @test_skip   Trixi.run("../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml") # errors for me
  @test_nowarn Trixi.run("../examples/parameters_source_terms.toml")
  @test_nowarn Trixi.run("../examples/parameters_vortex.toml")
  @test_nowarn Trixi.run("../examples/parameters_vortex_split_shockcapturing.toml")
  @test_nowarn Trixi.run("../examples/parameters_weak_blast_wave_shockcapturing.toml")
end

# Only run extended tests if environment variable is set
if haskey(ENV, "TRIXI_TEST_EXTENDED") && lowercase(ENV["TRIXI_TEST_EXTENDED"]) in ("1", "on", "yes")
  @testset "Examples (long execution time)" begin
    @test_nowarn Trixi.run("../examples/parameters_blob.toml")
    @test_nowarn Trixi.run("../examples/parameters_blob_amr.toml")
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
