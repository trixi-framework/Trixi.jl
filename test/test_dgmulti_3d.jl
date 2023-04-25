module TestExamplesDGMulti3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "dgmulti_3d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 3D" begin
  # 3d tet/hex tests
  @trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.0010029534292051608, 0.0011682205957721673, 0.001072975385793516, 0.000997247778892257, 0.0039364354651358294] ./ sqrt(8),
      linf = [0.003660737033303718, 0.005625620600749226, 0.0030566354814669516, 0.0041580358824311325, 0.019326660236036464]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.014932088450136542, 0.017080219613061526, 0.016589517840793006, 0.015905000907070196, 0.03903416208587798] ./ sqrt(8),
      linf = [0.06856547797256729, 0.08225664880340489, 0.06925055630951782, 0.06913016119820181, 0.19161418499621874]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Hexahedral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      element_type = Hex(),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.00030580190715769566, 0.00040146357607439464, 0.00040146357607564597, 0.000401463576075708, 0.0015749412434154315] ./ sqrt(8),
      linf = [0.00036910287847780054, 0.00042659774184228283, 0.0004265977427213574, 0.00042659774250686233, 0.00143803344597071]
    )
  end

  @trixi_testset "elixir_euler_curved.jl (Hex elements, SBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
      l2 = [0.018354883045936066, 0.024412704052042846, 0.024408520416087945, 0.01816314570880129, 0.039342805507972006],
      linf = [0.14862225990775757, 0.28952368161864683, 0.2912054484817035, 0.1456603133854122, 0.3315354586775472]
    )
  end

  @trixi_testset "elixir_euler_curved.jl (Hex elements, GaussSBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
      approximation_type=GaussSBP(),
      l2 = [0.002631131519508634, 0.0029144224044954105, 0.002913889110662827, 0.002615140832314194, 0.006881528610614373],
      linf = [0.020996114874140215, 0.021314522450134543, 0.021288322783006297, 0.020273381695435244, 0.052598740390024545]
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.0010317074322517949, 0.0012277090547035293, 0.0011273991123913515, 0.0010418496196130177, 0.004058878478404962] ./ sqrt(8),
      linf = [0.003227752881827861, 0.005620317864620361, 0.0030514833972379307, 0.003987027618439498, 0.019282224709831652]
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl (Hexahedral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      element_type = Hex(),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.00034230612468547436, 0.00044397204714598747, 0.0004439720471461567, 0.0004439720471464591, 0.0016639410646990126] ./ sqrt(8),
      linf = [0.0003674374460325147, 0.0004253921341716982, 0.0004253921340786615, 0.0004253921340831024, 0.0014333414071048267]
    )
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl (Hexahedral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      element_type = Hex(),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
      l2 = [0.001712443468716032, 0.002491315550718859, 0.0024913155507195303, 0.002491315550720031, 0.008585818982343299] ./ sqrt(8),
      linf = [0.003810078279323559, 0.004998778644230928, 0.004998778643986235, 0.0049987786444081195, 0.016455044373650196]
    )
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      polydeg = 3, tspan = (0.0, 1.0), cells_per_dimension = (2, 2, 2),
      l2   = [0.0003612827827560599, 0.06219350883951729, 0.062193508839503864, 0.08121963221634831, 0.07082703570808184],
      linf = [0.0007893509649821162, 0.1481953939988877, 0.14819539399791176, 0.14847291108358926, 0.21313533492212855]
      )
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl (GaussSBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      polydeg = 3, approximation_type = GaussSBP(), tspan = (0.0, 1.0), cells_per_dimension = (2, 2, 2),
      l2 = [0.00036128278275524326, 0.062193508839511434, 0.06219350883949677, 0.08121963221635205, 0.07082703570765223],
      linf = [0.000789350964946367, 0.14819539399525805, 0.14819539399590542, 0.14847291107658706, 0.21313533492059378]
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
      l2 = [0.0024092707138829925, 0.003318758964118284, 0.0033187589641182386, 0.003318758964118252, 0.012689348410504253],
      linf = [0.006118565824207778, 0.008486456080185167, 0.008486456080180282, 0.008486456080185611, 0.035113544599208346]
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
      l2 = [0.0034543609010604407, 0.004944363692625066, 0.0049443636926250435, 0.004944363692625037, 0.01788695279620914],
      linf = [0.013861851418853988, 0.02126572106620328, 0.021265721066209053, 0.021265721066210386, 0.0771455289446683]
    )
  end

  @trixi_testset "elixir_euler_fdsbp_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
      l2 = [7.561896970325353e-5, 6.884047859361093e-5, 6.884047859363204e-5, 6.884047859361148e-5, 0.000201107274617457],
      linf = [0.0001337520020225913, 0.00011571467799287305, 0.0001157146779990903, 0.00011571467799376123, 0.0003446082308800058]
    )
  end

end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
