module TestExamplesP4estMesh3D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "p4est_3d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "P4estMesh3D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [0.00016263963870641478], 
      linf = [0.0014537194925779984])
  end

  @trixi_testset "elixir_advection_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_curved.jl"),
      l2   = [0.0004750004258546538], 
      linf = [0.026527551737137167])
  end

  @trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
      l2   = [0.00253595715323843], 
      linf = [0.016486952252155795])
  end

  @trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [9.773852895157622e-6],
      linf = [0.0005853874124926162])
  end

  @trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_curved.jl"),
      l2   = [3.28458370046859e-5],
      linf = [0.003607158327703943])
  end

  @trixi_testset "elixir_advection_cubed_sphere.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
      l2   = [0.002006918015656413], 
      linf = [0.027655117058380085])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.002590388934758452], 
      linf = [0.01840757696885409])
  end

  @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
      l2   = [0.0002509450739533392, 0.00023785492442943794, 0.0002846031670226135, 0.0002932332754336982, 0.0006288429844253074], 
      linf = [0.011596300191441644, 0.009930481243511924, 0.013095959848100192, 0.02014760183990494, 0.03885749726552046])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [0.0015695663270391823, 0.0015490919943862824, 0.0015490919943864445, 0.0015490919943865404, 0.0030142321185958622], 
      linf = [0.011169568009149922, 0.012122645263175524, 0.012122645263178411, 0.012122645263167309, 0.022766806484090463])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [7.853107070208087e-15, 4.3176689497299765e-14, 4.605888682511052e-14, 5.468131071212009e-14, 9.42558567776032e-14],
      linf = [1.0187406473960436e-12, 4.899927685819705e-12, 6.2305716141963785e-12, 7.49544870615182e-12, 1.149125239408022e-11],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_euler_free_stream_extruded.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_extruded.jl"),
      l2   = [1.6667578354338437e-14, 8.351207753024527e-15, 1.805859250264852e-14, 1.90966219766217e-14, 1.90966219766217e-14],
      linf = [1.0413891970983968e-13, 4.988787161153141e-13, 2.0694557179012918e-13, 6.221689829999377e-13, 1.0587086762825493e-12])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.00921799151005215, 0.007057841476498664, 0.0074046565631184, 0.007421119519873141, 0.023272322544764468],
      linf = [0.18671575807969953, 0.2550156016690984, 0.2577539185993992, 0.26308798001518957, 0.443547750485219])
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [7.82070951e-02, 4.33260474e-02, 4.33260474e-02, 4.33260474e-02, 3.75260911e-01],
      linf = [7.45329845e-01, 3.21754792e-01, 3.21754792e-01, 3.21754792e-01, 4.76151527e+00],
      tspan = (0.0, 0.3))
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
