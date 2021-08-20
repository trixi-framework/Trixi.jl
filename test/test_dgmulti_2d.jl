module TestExamplesDGMulti2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "dgmulti_2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 2D" begin
  @trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      l2 = [0.0013463253573454783, 0.0014235911638070984, 0.0014235911638076622, 0.004721923810347086],
      linf = [0.0015248269221803668, 0.002070690855385582, 0.002070690855385804, 0.004913338290754687]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      approximation_type = SBP(),
      l2 = [0.007465048853079056, 0.005297478943323888, 0.005297478943334149, 0.014701766352428895],
      linf = [0.02149035033755564, 0.0135293837651278, 0.013529383764659286, 0.03269081948752639]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      element_type = Quad(),
      l2 = [0.0002909691660845978, 0.0002811425883546657, 0.0002811425883549579, 0.0010200600240538172],
      linf = [0.0004970396373780162, 0.0004059109438805386, 0.00040591094388231497, 0.0014247618507141624]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (EC) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      l2 = [0.008107539211003132, 0.007464472445092778, 0.007464472445093055, 0.01597648138530006],
      linf = [0.012298218434060981, 0.013874789519390918, 0.013874789519420005, 0.040393065744379175]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.012858228819248307, 0.010649745431713896, 0.010649745431680024, 0.02628727578633061],
      linf = [0.03733928157930677, 0.053088127555369624, 0.05308812755616854, 0.13379093830601718]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      element_type = Quad(),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.0029320668218228204, 0.0030626480847183927, 0.0030626480847164876, 0.0068440872356226165],
      linf = [0.013699485223367613, 0.012808498911205168, 0.01280849891121405, 0.022408991793810618]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"), 2)
    @test isapprox(mean_convergence[:l2], [4.2498632232077815, 4.133717042428824, 4.133717042041395, 4.086223177072245], rtol=0.05)
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      l2 = [0.0014986508075708323, 0.001528523420746786, 0.0015285234207473158, 0.004846505183839211],
      linf = [0.0015062108658376872, 0.0019373508504645365, 0.0019373508504538783, 0.004742686826709086]
    )
  end

  @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangulate_pkg_mesh.jl"),
      l2 = [4.664661209491976e-6, 3.7033509525940745e-6, 4.794877426562555e-6, 1.2682723101532175e-5],
      linf = [2.5099852761334418e-5, 2.2683684021362893e-5, 2.6180448559287584e-5, 5.5752932611508044e-5]
    )
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
