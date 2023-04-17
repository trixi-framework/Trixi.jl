module TestExamples2DLatticeBoltzmann

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Lattice-Boltzmann" begin
  @trixi_testset "elixir_lbm_constant.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_constant.jl"),
      l2   = [4.888991832247047e-15, 4.8856380534982224e-15, 5.140829677785587e-16,
              7.340293204570167e-16, 2.0559494114924474e-15, 6.125746684189216e-16,
              1.6545443003155128e-16, 6.001333022242579e-16, 9.450994018139234e-15],
      linf = [5.551115123125783e-15, 5.662137425588298e-15, 1.2212453270876722e-15,
              1.27675647831893e-15, 2.4980018054066022e-15, 7.494005416219807e-16,
              4.3021142204224816e-16, 8.881784197001252e-16, 1.0436096431476471e-14])
  end

  @trixi_testset "elixir_lbm_couette.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_couette.jl"),
      l2   = [0.0007899749117603378, 7.0995283148275575e-6, 0.0007454191223764233,
              1.6482025869100257e-5, 0.00012684365365448903, 0.0001198942846383015,
              0.00028436349827736705, 0.0003005161103138576, 4.496683876631818e-5],
      linf = [0.005596384769998769, 4.771160474496827e-5, 0.005270322068908595,
              0.00011747787108790098, 0.00084326349695725, 0.000795551892211168,
              0.001956482118303543, 0.0020739599893902436, 0.00032606270109525326],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_lbm_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_lid_driven_cavity.jl"),
      l2   = [0.0013628495945172754, 0.00021475256243322154, 0.0012579141312268184,
              0.00036542734715110765, 0.00024127756258120715, 0.00022899415795341014,
              0.0004225564518328741, 0.0004593854895507851, 0.00044244398903669927],
      linf = [0.025886626070758242, 0.00573859077176217, 0.027568805277855102, 0.00946724671122974,
              0.004031686575556803, 0.0038728927083346437, 0.020038695575169005,
              0.02061789496737146, 0.05568236920459335],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_lbm_couette.jl with initial_condition_couette_steady" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_couette.jl"),
      l2   = [9.321369073400123e-16, 1.6498793963435488e-6, 5.211495843124065e-16,
              1.6520893954826173e-6, 1.0406056181388841e-5, 8.801606429417205e-6,
              8.801710065560555e-6, 1.040614383799995e-5, 2.6135657178357052e-15],
      linf = [1.4432899320127035e-15, 2.1821189867266e-6, 8.881784197001252e-16,
              2.2481261510165496e-6, 1.0692966335143494e-5, 9.606391697600247e-6,
              9.62138334279633e-6, 1.0725969916147021e-5, 3.3861802251067274e-15],
      initial_condition=function initial_condition_couette_steady(x, t, equations::LatticeBoltzmannEquations2D)
        # Initial state for a *steady* Couette flow setup. To be used in combination with
        # [`boundary_condition_couette`](@ref) and [`boundary_condition_noslip_wall`](@ref).
        @unpack L, u0, rho0 = equations

        rho = rho0
        v1 = u0 * x[2] / L
        v2 = 0

        return equilibrium_distribution(rho, v1, v2, equations)
        end,
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_lbm_lid_driven_cavity.jl with stationary walls" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_lid_driven_cavity.jl"),
      l2   = [1.7198203373689985e-16, 1.685644347036533e-16, 2.1604974801394525e-16,
              2.1527076266915764e-16, 4.2170298143732604e-17, 5.160156233016299e-17,
              6.167794865198169e-17, 5.24166554417795e-17, 6.694740573885739e-16],
      linf = [5.967448757360216e-16, 6.522560269672795e-16, 6.522560269672795e-16,
              6.245004513516506e-16, 2.1163626406917047e-16, 2.185751579730777e-16,
              2.185751579730777e-16, 2.393918396847994e-16, 1.887379141862766e-15],
      boundary_conditions=boundary_condition_noslip_wall,
      tspan = (0, 0.1))
  end
end

end # module
