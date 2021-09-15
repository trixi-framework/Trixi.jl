module TestExamplesThreaded

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

@testset "Threaded tests" begin
  @testset "TreeMesh" begin
    @trixi_testset "elixir_advection_restart.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_restart.jl"),
        l2   = [1.2148032444677485e-5],
        linf = [6.495644794757283e-5])
    end

    @trixi_testset "elixir_advection_amr_refine_twice.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr_refine_twice.jl"),
        l2   = [0.00019847333806230843],
        linf = [0.005591345460895569])
    end

    @trixi_testset "elixir_advection_amr_coarsen_twice.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr_coarsen_twice.jl"),
        l2   = [0.00519897841357112],
        linf = [0.06272325552264647])
    end

    @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
        l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
        linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5],
        rtol = 0.001)
    end

    @trixi_testset "elixir_euler_ec.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"),
        l2   = [0.061751715597716854, 0.05018223615408711, 0.05018989446443463, 0.225871559730513],
        linf = [0.29347582879608825, 0.31081249232844693, 0.3107380389947736, 1.0540358049885143])
    end
  end


  @testset "StructuredMesh" begin
    @trixi_testset "elixir_advection_restart.jl with waving flag mesh" begin
      @test_trixi_include(joinpath(examples_dir(), "structured_2d_dgsem", "elixir_advection_restart.jl"),
        l2   = [0.00017274040834067234],
        linf = [0.0015435741643734513],
        elixir_file="elixir_advection_waving_flag.jl",
        restart_file="restart_000041.h5")
    end

    @trixi_testset "elixir_mhd_ec.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "structured_2d_dgsem", "elixir_mhd_ec.jl"),
        l2   = [0.04937480811868297, 0.06117033019988596, 0.060998028674664716, 0.03155145889799417,
                0.2319175391388658, 0.02476283192966346, 0.024483244374818587, 0.035439957899127385,
                0.0016022148194667542],
        linf = [0.24749024430983746, 0.2990608279625713, 0.3966937932860247, 0.22265033744519683,
                0.9757376320946505, 0.12123736788315098, 0.12837436699267113, 0.17793825293524734,
                0.03460761690059514],
        tspan = (0.0, 0.3))
    end
  end


  @testset "UnstructuredMesh" begin
    @trixi_testset "elixir_ape_gauss_wall.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_dgsem", "elixir_ape_gauss_wall.jl"),
        l2   = [0.029331247985489625, 0.02934616721732521, 0.03803253571320854, 0.0,
                7.465932985352019e-16, 1.4931865970704038e-15, 1.4931865970704038e-15],
        linf = [0.3626825396196784, 0.3684490307932018, 0.8477478712580901, 0.0,
                8.881784197001252e-16, 1.7763568394002505e-15, 1.7763568394002505e-15],
        tspan = (0.0, 5.0))
    end
  end


  @testset "P4estMesh" begin
    @trixi_testset "elixir_euler_source_terms_nonperiodic_unstructured.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_euler_source_terms_nonperiodic_unstructured.jl"),
        l2   = [0.005689496253354319, 0.004522481295923261, 0.004522481295922983, 0.009971628336802528],
        linf = [0.05125433503504517, 0.05343803272241532, 0.053438032722404216, 0.09032097668196482])
    end

    @trixi_testset "elixir_eulergravity_convergence.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_eulergravity_convergence.jl"),
        l2   = [0.00024871265138964204, 0.0003370077102132591, 0.0003370077102131964, 0.0007231525513793697],
        linf = [0.0015813032944647087, 0.0020494288423820173, 0.0020494288423824614, 0.004793821195083758],
        tspan = (0.0, 0.1))
    end
  end


  @testset "DGMulti" begin
    @trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
      @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_euler_weakform.jl"),
        cells_per_dimension = (4, 4),
        volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
        surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
        approximation_type = SBP(),
        l2 = [0.01285864081726596, 0.010650165503847099, 0.01065016550381281, 0.026286162111579015],
        linf = [0.037333313274372504, 0.05308320130762212, 0.05308320130841948, 0.13378665881805185],
      )
    end

    @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_euler_triangulate_pkg_mesh.jl"),
        l2 = [4.664661209491976e-6, 3.7033509525940745e-6, 4.794877426562555e-6, 1.2682723101532175e-5],
        linf = [2.5099852761334418e-5, 2.2683684021362893e-5, 2.6180448559287584e-5, 5.5752932611508044e-5]
      )
    end
  end
end

# Clean up afterwards: delete Trixi output directory
Trixi.mpi_isroot() && isdir(outdir) && @test_nowarn rm(outdir, recursive=true)

end # module
