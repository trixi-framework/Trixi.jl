module TestExamples2DTri

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "simplicial_2d_dg")

@testset "2D simplicial mesh tests" begin
  @trixi_testset "elixir_euler_triangular_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh.jl"),
      l2 = [0.0006456213233232335, 0.0012103605920880222, 0.0012103605920879606, 0.004127251610067376],
      linf = [0.0007930980554167189, 0.0021736528649873854, 0.0021736528649864972, 0.005871873927952187]
    )
  end

  @trixi_testset "elixir_euler_periodic_triangular_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic_triangular_mesh.jl"),
      l2 = [0.000672767625167618, 0.001269847226694384, 0.0012698472266944028, 0.004177337699476258],
      linf = [0.0007498340156470995, 0.0021716152870556726, 0.002171615287057893, 0.005873584107397356]
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
      l2 = [0.0001016711916046445, 0.00011071422293274785, 0.00011212482087451142, 0.00035893791736543447],
      linf = [0.00035073781634786805, 0.00039815763002271076, 0.00041642100745109545, 0.0009481311054404529]
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
      l2 = [0.00015946119543800883, 0.000264475476940273, 0.00026447547694017894, 0.0008996889622631928],
      linf = [0.0002365475762040603, 0.00032922335054608176, 0.0003292233505463038, 0.0012473538592372435]
    )
  end
end

@testset "2D simplicial flux differencing" begin
  @trixi_testset "elixir_euler_triangular_mesh_flux_diff.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh_flux_diff.jl"),
      l2 = [0.0023756965829858814, 0.002810713803967761, 0.0028107138039682894, 0.008040750705578132],
      linf = [0.004397202318891624, 0.00446064886127262, 0.004460648861263294, 0.012346721614063583]
    )
  end

  @trixi_testset "elixir_euler_sbp_triangular_mesh_flux_diff.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sbp_triangular_mesh_flux_diff.jl"),
      l2 = [0.0019514080401385394, 0.00341875398076509, 0.0034187539807649155, 0.011829447744071755],
      linf = [0.004078990424579931, 0.007257990068658904, 0.007257990068565867, 0.027380769039135444]
    )
  end
end

end # module
