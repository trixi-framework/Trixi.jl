module TestExamples2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "paper-self-gravitating-gas-dynamics")

@testset "2D" begin

# Run basic tests
@testset "paper-self-gravitating-gas-dynamics" begin
  @testset "taal-confirmed elixir_euler_eoc_test.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_eoc_test.jl"),
      l2   = [0.00017409779099463607, 0.0003369287450282371, 0.00033692874502819616, 0.0006099035183426747],
      linf = [0.0010793454782482836, 0.0018836374478419238, 0.0018836374478410356, 0.003971446179607874])
  end

  @testset "taal-confirmed elixir_euler_eoc_test.jl with polydeg=4" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_eoc_test.jl"),
      l2   = [1.7187032983384504e-5, 2.6780178144541376e-5, 2.678017814469407e-5, 4.952410417693103e-5],
      linf = [0.00015018092862240096, 0.00016548331714294484, 0.00016548331714405506, 0.00043726245511699346],
      polydeg = 4)
  end


  @testset "taal-confirmed elixir_hyp_diff_eoc_test.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_eoc_test.jl"),
      l2   = [0.00315402168051244, 0.012394424055283394, 0.021859728673870843],
      linf = [0.017332075119072865, 0.07843510773347322, 0.11325788389718668])
  end

  @testset "taal-confirmed elixir_hyp_diff_eoc_test.jl with polydeg=4" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_eoc_test.jl"),
    l2   = [0.00025112830138292663, 0.0008808243851096586, 0.0016313343234903468],
    linf = [0.001719090967553516, 0.0031291844657076145, 0.00994609342322228],
    polydeg = 4)
  end


  @testset "taal-confirmed elixir_eulergravity_eoc_test.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_eoc_test.jl"),
      l2   = [0.0002487158370511598, 0.0003370291440916084, 0.00033702914409161063, 0.0007231934514459757],
      linf = [0.001581173125044355, 0.002049389755695241, 0.0020493897556961294, 0.004793721268126383],
      tspan = (0.0, 0.1))
  end

  @testset "taal-confirmed elixir_eulergravity_eoc_test.jl with polydeg=4" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_eoc_test.jl"),
      l2   = [1.9536732064098693e-5, 2.756381055173374e-5, 2.7563810551703437e-5, 5.688705902953846e-5],
      linf = [0.00012335977351507488, 0.00020086338378089152, 0.00020086338378044744, 0.0004962132679873221],
      tspan = (0.0, 0.1), polydeg = 4)
  end

  @testset "taal-confirmed elixir_eulergravity_eoc_test.jl with 1st order RK3S*" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_eoc_test.jl"),
      l2   = [0.00024871583705119436, 0.0003370291440915927, 0.0003370291440916112, 0.0007231934514459859],
      linf = [0.001581173125044355, 0.002049389755695241, 0.0020493897556961294, 0.004793721268126383],
      tspan = (0.0, 0.1), timestep_gravity=Trixi.timestep_gravity_erk51_3Sstar!)
  end

  @testset "taal-confirmed elixir_eulergravity_eoc_test.jl with 3rd order RK3S*" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_eoc_test.jl"),
      l2   = [0.000248715837047647, 0.0003370291440257414, 0.00033702914402587556, 0.0007231934513057375],
      linf = [0.00158117312532835, 0.0020493897540796446, 0.0020493897540800887, 0.0047937212650124295],
      tspan = (0.0, 0.1), timestep_gravity=Trixi.timestep_gravity_erk53_3Sstar!)
  end


  @testset "taal-check-me elixir_eulergravity_jeans_instability.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_jeans_instability.jl"),
      l2   = [10733.634574440104, 13356.777246273672, 1.9930894028451876e-6, 26834.07879379781],
      linf = [15194.297536645085, 18881.47693900588, 8.325325156694497e-6, 37972.99978450313],
      tspan = (0.0, 0.1))
  end

  @testset "taal-check-me elixir_eulergravity_jeans_instability.jl with RK3S*" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_jeans_instability.jl"),
      l2   = [10734.59878993429, 13358.214052395579, 2.7246732181080924e-6, 26836.489332980615],
      linf = [15195.661845114082, 18883.507539561684, 1.1096401891274226e-5, 37976.4105797708],
      tspan = (0.0, 0.1),
      parameters=ParametersEulerGravity(background_density=1.5e7,
                                        gravitational_constant=6.674e-8,
                                        # FIXME Taal restore after Taam sync
                                        cfl=1.2,
                                        n_iterations_max=1000,
                                        timestep_gravity=timestep_gravity_erk52_3Sstar!))
  end


  @testset "taal-confirmed cfl-magic elixir_eulergravity_sedov_blast_wave.jl" begin
    # Reducing the CFL number reduces the differences between Taam and Taal.
    # Using Trixi.solve(ode, Trixi.CarpenterKennedy2N54() instead of
    # solve(ode, CarpenterKennedy2N54(williamson_condition=false)
    # reduces the differences even further (often a factor of ca. 2).
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_sedov_blast_wave.jl"),
      l2   = [0.04630745182870653, 0.06507397069667138, 0.06507397069667123, 0.48971269294890085],
      linf = [2.3861430058270847, 4.083635578775231, 4.083635578775232, 16.246070713311475],
      tspan = (0.0, 0.05))
  end

  @testset "taal-confirmed cfl-magic elixir_eulergravity_sedov_blast_wave.jl with amr_interval=0 and ref-level=8" begin
    # Same comments as for `elixir_eulergravity_sedov_blast_wave.jl` with default settings
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_sedov_blast_wave.jl"),
      l2   = [0.0028922121586238323, 0.013724796675028317, 0.013724796675028307, 0.05822941648860658],
      linf = [0.26747911779347966, 1.385822018653034, 1.385822018653034, 4.071204772447614],
      tspan = (0.0, 0.005), initial_refinement_level=8, amr_callback=TrivialCallback())
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D

end #module
