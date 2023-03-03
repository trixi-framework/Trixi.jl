module TestExamples2DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Shallow Water" begin
  @trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [0.9911807999456177, 0.7340358820001013, 0.7446688375175905, 0.5875351036989047],
      linf = [2.0118414065026426, 2.9946841579032526, 2.6555844309739545, 3.0],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [0.9130579602987144, 1.0602847041965408e-14, 1.082225645390032e-14, 0.9130579602987147],
      linf = [2.113062037615659, 4.6613606802974e-14, 5.4225772771633196e-14, 2.1130620376156584],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_wall.jl"),
      l2   = [0.9130579602987144, 1.0602847041965408e-14, 1.082225645390032e-14, 0.9130579602987147],
      linf = [2.113062037615659, 4.6613606802974e-14, 5.4225772771633196e-14, 2.1130620376156584],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [0.9130579602987147, 9.68729463970494e-15, 9.694538537436981e-15, 0.9130579602987147],
      linf = [2.1130620376156584, 2.3875905654916432e-14, 2.2492839032269154e-14, 2.1130620376156584],
      surface_flux=(FluxHydrostaticReconstruction(flux_lax_friedrichs, hydrostatic_reconstruction_audusse_etal), flux_nonconservative_audusse_etal),
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced_wet_dry.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_wet_dry.jl"),
      l2   = [0.03904507146870548, 2.0953804074196604e-14, 5.139180449007378e-17, 0.1649319748280072],
      linf = [0.49999999999989153, 4.967300398427917e-14, 4.841253465352248e-16, 1.9999999999999991],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.001868474306068482, 0.01731687445878443, 0.017649083171490863, 6.274146767717023e-5],
      linf = [0.016962486402209986, 0.08768628853889782, 0.09038488750767648, 0.0001819675955490041],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms_dirichlet.jl"),
      l2   = [0.0018746929418489125, 0.017332321628469628, 0.01634953679145536, 6.274146767717023e-5],
      linf = [0.016262353691956388, 0.08726160620859424, 0.09043621801418844, 0.0001819675955490041],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0018957692481057034, 0.016943229710439864, 0.01755623297390675, 6.274146767717414e-5],
      linf = [0.015156105797771602, 0.07964811135780492, 0.0839787097210376, 0.0001819675955490041],
      tspan = (0.0, 0.025), surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal))
  end

  @trixi_testset "elixir_shallowwater_conical_island.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_conical_island.jl"),
        l2   = [0.26189905898488786, 0.13109288589313403, 0.1310928914335753, 0.0011537702354532694],
        linf = [1.1129921709644124, 0.22929803337323118, 0.22929795378591095, 0.021790250683516282],
        tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_parabolic_bowl.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_parabolic_bowl.jl"),
      l2   = [0.009551029879419562, 0.0022746779961813663, 0.006093909535093138, 3.649315951724414e-17],
      linf = [0.037838314224258915, 0.008905881822753982, 0.02385742969303331, 2.7755575615628914e-16],
      tspan = (0.0, 0.25), 
      basis = LobattoLegendreBasis(3))
  end
end

end # module
