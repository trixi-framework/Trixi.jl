module TestExamplesThreaded

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

@testset "Threaded tests" begin
  @testset "TreeMesh" begin
    @trixi_testset "elixir_advection_restart.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_restart.jl"),
        # Expected errors are exactly the same as in the serial test!
        l2   = [7.81674284320524e-6],
        linf = [6.314906965243505e-5])
    end

    @trixi_testset "elixir_advection_amr_refine_twice.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr_refine_twice.jl"),
        l2   = [0.00020547512522578292],
        linf = [0.007831753383083506])
    end

    @trixi_testset "elixir_advection_amr_coarsen_twice.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr_coarsen_twice.jl"),
        l2   = [0.0014321062757891826],
        linf = [0.0253454486893413])
    end

    @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
        l2   = [2.259440511766445e-6, 2.318888155713922e-6, 2.3188881557894307e-6, 6.3327863238858925e-6],
        linf = [1.498738264560373e-5, 1.9182011928187137e-5, 1.918201192685487e-5, 6.0526717141407005e-5],
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
        l2   = [0.00016265538265929818],
        linf = [0.0015194252169410394],
        rtol = 5.0e-5, # Higher tolerance to make tests pass in CI (in particular with macOS)
        elixir_file="elixir_advection_waving_flag.jl",
        restart_file="restart_000021.h5")
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
    @trixi_testset "elixir_acoustics_gauss_wall.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_dgsem", "elixir_acoustics_gauss_wall.jl"),
      l2   = [0.029330394861252995, 0.029345079728907965, 0.03803795043486467, 0.0,
              7.175152371650832e-16, 1.4350304743301665e-15, 1.4350304743301665e-15],
      linf = [0.36236334472179443, 0.3690785638275256, 0.8475748723784078, 0.0,
              8.881784197001252e-16, 1.7763568394002505e-15, 1.7763568394002505e-15],
        tspan = (0.0, 5.0))
    end
  end


  @testset "P4estMesh" begin
    @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
        l2   = [0.0034516244508588046, 0.0023420334036925493, 0.0024261923964557187, 0.004731710454271893],
        linf = [0.04155789011775046, 0.024772109862748914, 0.03759938693042297, 0.08039824959535657])
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
        l2 = [0.006400337855843578, 0.005303799804137764, 0.005303799804119745, 0.013204169007030144],
        linf = [0.03798302318566282, 0.05321027922532284, 0.05321027922605448, 0.13392025411839015],
      )
    end

    @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_euler_triangulate_pkg_mesh.jl"),
        l2 = [2.344080455438114e-6, 1.8610038753097983e-6, 2.4095165666095305e-6, 6.373308158814308e-6],
        linf = [2.5099852761334418e-5, 2.2683684021362893e-5, 2.6180448559287584e-5, 5.5752932611508044e-5]
      )
    end

    @trixi_testset "elixir_euler_fdsbp_periodic.jl" begin
      @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_euler_fdsbp_periodic.jl"),
        l2 = [1.3333320340010056e-6, 2.044834627970641e-6, 2.044834627855601e-6, 5.282189803559564e-6],
        linf = [2.7000151718858945e-6, 3.988595028259212e-6, 3.9885950273710336e-6, 8.848583042286862e-6]
      )
    end
  end
end

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && isdir(outdir) && @test_nowarn rm(outdir, recursive=true)

end # module
