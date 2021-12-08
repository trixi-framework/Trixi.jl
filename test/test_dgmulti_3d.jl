module TestExamplesDGMulti3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "dgmulti_3d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 3D" begin
  # 3d tet/hex tests
  @trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.0010029534292051608, 0.0011682205957721673, 0.001072975385793516, 0.000997247778892257, 0.0039364354651358294] ./ sqrt(8),
      linf = [0.003660737033303718, 0.005625620600749226, 0.0030566354814669516, 0.0041580358824311325, 0.019326660236036464] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      l2 = [0.014932088450136542, 0.017080219613061526, 0.016589517840793006, 0.015905000907070196, 0.03903416208587798] ./ sqrt(8),
      linf = [0.06856547797256729, 0.08225664880340489, 0.06925055630951782, 0.06913016119820181, 0.19161418499621874] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Hexahedral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      element_type = Hex(),
      l2 = [0.00030580190715769566, 0.00040146357607439464, 0.00040146357607564597, 0.000401463576075708, 0.0015749412434154315] ./ sqrt(8),
      linf = [0.00036910287847780054, 0.00042659774184228283, 0.0004265977427213574, 0.00042659774250686233, 0.00143803344597071] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      l2 = [0.0010317074322517949, 0.0012277090547035293, 0.0011273991123913515, 0.0010418496196130177, 0.004058878478404962] ./ sqrt(8),
      linf = [0.003227752881827861, 0.005620317864620361, 0.0030514833972379307, 0.003987027618439498, 0.019282224709831652] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl (Hexahedral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      element_type = Hex(),
      l2 = [0.00034230612468547436, 0.00044397204714598747, 0.0004439720471461567, 0.0004439720471464591, 0.0016639410646990126] ./ sqrt(8),
      linf = [0.0003674374460325147, 0.0004253921341716982, 0.0004253921340786615, 0.0004253921340831024, 0.0014333414071048267] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl (Hexahedral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      element_type = Hex(),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.001712443468716032, 0.002491315550718859, 0.0024913155507195303, 0.002491315550720031, 0.008585818982343299] ./ sqrt(8),
      linf = [0.003810078279323559, 0.004998778644230928, 0.004998778643986235, 0.0049987786444081195, 0.016455044373650196] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      polydeg = 3, tspan = (0.0, 1.0), cells_per_dimension = (2, 2, 2),
      l2   = [0.005690062908500644, 0.9795235053774554, 0.97952350537847, 1.279177527342275, 1.1154981860062954] ./ sqrt(8),
      linf = [0.0007893509649821162, 0.1481953939988877, 0.14819539399791176, 0.14847291108358926, 0.21313533492212855] ./ sqrt(8),
      )
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl (GaussSBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      polydeg = 3, approximation_type = GaussSBP(), tspan = (0.0, 1.0), cells_per_dimension = (2, 2, 2),
      l2 = [0.005690062908496233, 0.9795235053781463, 0.9795235053778316, 1.2791775273427364, 1.1154981860090258] ./ sqrt(8),
      linf = [0.000789350964946367, 0.14819539399525805, 0.14819539399590542, 0.14847291107658706, 0.21313533492059378] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl (FD SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      element_type = Hex(),
      cells_per_dimension = (2, 2, 2),
      approximation_type = derivative_operator(
        SummationByPartsOperators.MattssonNordström2004(),
        derivative_order=1, accuracy_order=2,
        xmin=0.0, xmax=1.0, N=8),
      l2 = [0.006814361236327181, 0.009386715771140851, 0.0093867157711408, 0.00938671577114083, 0.0358906753069569] ./ sqrt(8),
      linf = [0.006118848678484001, 0.008486842368975012, 0.008486842368970349, 0.008486842368975456, 0.03511617898229247] ./ sqrt(8)
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl (FD SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      element_type = Hex(),
      cells_per_dimension = (2, 2, 2),
      approximation_type = derivative_operator(
        SummationByPartsOperators.MattssonNordström2004(),
        derivative_order=1, accuracy_order=2,
        xmin=0.0, xmax=1.0, N=8),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      l2 = [0.009769348117265823, 0.013983224512102375, 0.01398322451210242, 0.013983224512102474, 0.050588386809931433] ./ sqrt(8),
      linf = [0.013859882019255698, 0.021262483438011515, 0.02126248343800885, 0.0212624834380013, 0.07713700592240169] ./ sqrt(8)
    )
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
