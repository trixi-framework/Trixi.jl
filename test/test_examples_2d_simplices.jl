module TestExamples2DTri

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "simplicial_2d_dg")

@testset "2D simplicial mesh tests" begin
  @trixi_testset "elixir_euler_triangular_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh.jl"),
      l2 = [0.0013463253573454718, 0.0014235911638071127, 0.0014235911638076826, 0.00472192381034704],
      linf = [0.0015248269221774802, 0.0020706908553849157, 0.0020706908553842496, 0.004913338290754243]
    )
  end

  @trixi_testset "elixir_euler_periodic_triangular_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic_triangular_mesh.jl"),
      l2 = [0.0014986508075708323, 0.001528523420746786, 0.0015285234207473158, 0.004846505183839211],
      linf = [0.0015062108658376872, 0.0019373508504645365, 0.0019373508504538783, 0.004742686826709086]
    )
  end

  @trixi_testset "elixir_ape_sbp_triangular_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_sbp_triangular_mesh.jl"),
      l2 = [0.13498182674793963, 0.13498182674793793, 0.10409926243751662, 0.0, 0.0, 0.0, 0.0],
      linf = [0.326623183281191, 0.3266231832808133, 0.349817130019491, 0.0, 0.0, 0.0, 0.0]
    )
  end

  @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangulate_pkg_mesh.jl"),
      l2 = [0.00015405739868423074, 0.00014530283464035389, 0.00014870936695617315, 0.00044410650633679334],
      linf = [0.00039269979059053384, 0.0004237090504681795, 0.000577877861525522, 0.0015066603278119928]
    )
  end

  @trixi_testset "2D simplicial convergence tests" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh_convergence.jl"), 2)
    @test isapprox(mean_convergence[:l2], [4.116645204366779, 4.06608993434891, 4.066089934205002, 4.114554671587996], rtol=0.05)
  end

end

@testset "2D quadrilateral tests (using simplicial DG code)" begin
  @trixi_testset "elixir_euler_quadrilateral_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_quadrilateral_mesh.jl"),
      l2 = [0.0002909691660845978, 0.0002811425883546657, 0.0002811425883549579, 0.0010200600240538172],
      linf = [0.0004970396373780162, 0.0004059109438805386, 0.00040591094388231497, 0.0014247618507141624]
    )
  end
end

@testset "2D simplicial flux differencing" begin
  @trixi_testset "elixir_euler_triangular_mesh_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh_ec.jl"),
      l2 = [0.008107539211003132, 0.007464472445092778, 0.007464472445093055, 0.01597648138530006],
      linf = [0.012298218434060981, 0.013874789519390918, 0.013874789519420005, 0.040393065744379175]
    )
  end

  @trixi_testset "elixir_euler_sbp_triangular_mesh_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sbp_triangular_mesh_ec.jl"),
      l2 = [0.012858228819248307, 0.010649745431713896, 0.010649745431680024, 0.02628727578633061],
      linf = [0.03733928157930677, 0.053088127555369624, 0.05308812755616854, 0.13379093830601718]
    )
  end
end

end # module
