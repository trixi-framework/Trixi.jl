
using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "Compressible Euler" begin
  @testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [2.2527950196212703e-8, 1.8187357193835156e-8, 7.705669939973104e-8],
      linf = [1.6205433861493646e-7, 1.465427772462391e-7, 5.372255111879554e-7])
  end

  @testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0011482554820185795, 0.00011482554830363504, 5.741277417754598e-6],
      linf = [0.004090978306814375, 0.00040909783135059663, 2.045489171820236e-5])
  end

  @testset "elixir_euler_density_wave.jl with initial_condition_density_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0037246420494099768, 0.0037246420494097534, 0.0018623210247056341],
      linf = [0.01853878721991542, 0.018538787219892106, 0.009269393609914633],
      initial_condition = initial_condition_density_pulse)
  end

  @testset "elixir_euler_density_wave.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [7.71293052584723e-16, 1.9712947511091717e-14, 7.50672833504266e-15],
      linf = [3.774758283725532e-15, 6.733502644351574e-14, 2.4868995751603507e-14],
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_euler_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
      l2   = [3.8099996914101204e-6, 1.6745575717106341e-6, 7.732189531480852e-6],
      linf = [1.2971473393186272e-5, 9.270328934274374e-6, 3.092514399671842e-5])
  end

  @testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.11908663558144943, 0.15491250781545182, 0.44523336050199985],
      linf = [0.27526364444744644, 0.27025123483580876, 0.9947524023443433])
  end

  @testset "elixir_euler_ec.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07909267609417114, 0.1018246500951966, 0.2959649187481973],
      linf = [0.23631829743146504, 0.2977756307879202, 0.8642794698697331],
      maxiters = 10,
      surface_flux = flux_shima_etal,
      volume_flux = flux_shima_etal)
  end

  @testset "elixir_euler_ec.jl with flux_ranocha" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07908920892644909, 0.10182280160297434, 0.2959098091388071],
      linf = [0.2366903058624107, 0.2982757330294113, 0.8649359933295631],
      maxiters = 10,
      surface_flux = flux_ranocha,
      volume_flux = flux_ranocha)
  end

  @testset "elixir_euler_ec.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07959780803600519, 0.10342491934977621, 0.2978851659149904],
      linf = [0.19228754121840885, 0.2524152253292552, 0.725604944702432],
      maxiters = 10,
      surface_flux = flux_hll,
      volume_flux = flux_ranocha)
  end

  @testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.11664633479399032, 0.1510307379005762, 0.4349674368602559],
      linf = [0.18712791503569948, 0.24590150230707294, 0.7035348754107953])
  end

  @testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.2526428105949643, 0.06850036315669353, 0.9265758557535149],
      linf = [2.987836852104084, 0.1706331519278507, 2.6658508877441838])
  end

  @testset "elixir_euler_sedov_blast_wave.jl with pressure" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.2983852642398792, 0.07961334122389492, 0.9270679298118678],
      linf = [3.185530212032373, 0.2136435252219927, 2.6651572539845736],
      shock_indicator_variable = pressure)
  end

  @testset "elixir_euler_sedov_blast_wave.jl with density" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.2796976935143354, 0.07078122470157631, 0.9274308941700738],
      linf = [3.1132929378257703, 0.17699767997572083, 2.66685149106963],
      shock_indicator_variable = density,
      cfl = 0.5)
  end

  @testset "elixir_euler_blast_wave.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [0.21535606561541146, 0.2805347806732941, 0.5589053506095726],
      linf = [1.508388610723991, 1.56220103779441, 2.0440877833210487],
      maxiters = 30)
  end
end
