module TestExamples1DShallowWater

# TODO: TrixiShallowWater: move any wet/dry tests to new package

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Shallow Water" begin
  @trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [0.244729018751225, 0.8583565222389505, 0.07330427577586297],
      linf = [2.1635021283528504, 3.8717508164234453, 1.7711213427919539],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_ec.jl with initial_condition_weak_blast_wave" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [0.39464782107209717, 2.03880864210846, 4.1623084150546725e-10],
      linf = [0.778905801278281, 3.2409883402608273, 7.419800190922032e-10],
      initial_condition=initial_condition_weak_blast_wave,
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [0.10416666834254829, 1.4352935256803184e-14, 0.10416666834254838],
      linf = [1.9999999999999996, 3.248036646353028e-14, 2.0],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [0.10416666834254835, 1.1891029971551825e-14, 0.10416666834254838],
      linf = [2.0000000000000018, 2.4019608337954543e-14, 2.0],
      surface_flux=(FluxHydrostaticReconstruction(flux_lax_friedrichs, hydrostatic_reconstruction_audusse_etal), flux_nonconservative_audusse_etal),
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced_wet_dry.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_wet_dry.jl"),
      l2   = [0.00965787167169024, 5.345454081916856e-14, 0.03857583749209928],
      linf = [0.4999999999998892, 2.2447689894899726e-13, 1.9999999999999714],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0022363707373868713, 0.01576799981934617, 4.436491725585346e-5],
      linf = [0.00893601803417754, 0.05939797350246456, 9.098379777405796e-5],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0022758146627220154, 0.015864082886204556, 4.436491725585346e-5],
      linf = [0.008457195427364006, 0.057201667446161064, 9.098379777405796e-5],
      tspan = (0.0, 0.025), surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal))
  end

  @trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms_dirichlet.jl"),
      l2   = [0.0022851099219788917, 0.01560453773635554, 4.43649172558535e-5],
      linf = [0.008934615705174398, 0.059403169140869405, 9.098379777405796e-5],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms_dirichlet.jl"),
      l2   = [0.0022956052733432287, 0.015540053559855601, 4.43649172558535e-5],
      linf = [0.008460440313118323, 0.05720939349382359, 9.098379777405796e-5],
      surface_flux=(FluxHydrostaticReconstruction(flux_hll, hydrostatic_reconstruction_audusse_etal), flux_nonconservative_audusse_etal),
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_well_balanced_nonperiodic.jl with Dirichlet boundary" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_nonperiodic.jl"),
      l2   = [1.725964362045055e-8, 5.0427180314307505e-16, 1.7259643530442137e-8],
      linf = [3.844551077492042e-8, 3.469453422316143e-15, 3.844551077492042e-8],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced_nonperiodic.jl with wall boundary" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_nonperiodic.jl"),
      l2   = [1.7259643614361866e-8, 3.5519018243195145e-16, 1.7259643530442137e-8],
      linf = [3.844551010878661e-8, 9.846474508971374e-16, 3.844551077492042e-8],
      tspan = (0.0, 0.25),
      boundary_condition = boundary_condition_slip_wall)
  end

  @trixi_testset "elixir_shallowwater_shock_capturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_shock_capturing.jl"),
      l2   = [0.07424140641160326, 0.2148642632748155, 0.0372579849000542],
      linf = [1.1209754279344226, 1.3230788645853582, 0.8646939843534251],
      tspan = (0.0, 0.05))
  end

  @trixi_testset "elixir_shallowwater_beach.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_beach.jl"),
      l2   = [0.17979210479598923, 1.2377495706611434, 6.289818963361573e-8],
      linf = [0.845938394800688, 3.3740800777086575, 4.4541473087633676e-7],
      tspan = (0.0, 0.05))
  end

  @trixi_testset "elixir_shallowwater_parabolic_bowl.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_parabolic_bowl.jl"),
      l2   = [8.965981683033589e-5, 1.8565707397810857e-5, 4.1043039226164336e-17],
      linf = [0.00041080213807871235, 0.00014823261488938177, 2.220446049250313e-16],
      tspan = (0.0, 0.05))
  end
end

end # module
