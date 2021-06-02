module TestExamples3DCurved

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

@testset "Curved mesh" begin
  @testset "elixir_advection_basic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_curved.jl"),
      l2   = [0.00013446460962856976],
      linf = [0.0012577781391462928])
  end

  @testset "elixir_advection_free_stream_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream_curved.jl"),
      l2   = [1.830875777528287e-14],
      linf = [7.491784970170556e-13],
      atol = 8e-13, # required to make tests pass on Windows
      )
  end

  @testset "elixir_advection_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic_curved.jl"),
      l2   = [6.522004549411137e-5],
      linf = [0.005554857853361295])
  end

  @testset "elixir_advection_restart_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_curved.jl"),
      l2   = [0.0281388160824776],
      linf = [0.08740635193023694])
  end

  @testset "elixir_euler_source_terms_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_curved.jl"),
      l2   = [0.01032310150257373, 0.009728768969448439, 0.009728768969448494, 0.009728768969448388, 0.015080412597559597],
      linf = [0.034894790428615874, 0.033835365548322116, 0.033835365548322116, 0.03383536554832034, 0.06785765131417065])
  end

  @testset "elixir_euler_free_stream_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_curved.jl"),
      l2   = [2.8815700334367128e-15, 9.361915278236651e-15, 9.95614203619935e-15, 1.6809941842374106e-14, 1.4815037041566735e-14],
      linf = [4.1300296516055823e-14, 2.0444756998472258e-13, 1.0133560657266116e-13, 3.1896707497480747e-13, 6.092903959142859e-13])
  end

  @testset "elixir_euler_free_stream_curved.jl with FluxRotated(flux_lax_friedrichs)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_curved.jl"),
      surface_flux=FluxRotated(flux_lax_friedrichs),
      l2   = [2.8815700334367128e-15, 9.361915278236651e-15, 9.95614203619935e-15, 1.6809941842374106e-14, 1.4815037041566735e-14],
      linf = [4.1300296516055823e-14, 2.0444756998472258e-13, 1.0133560657266116e-13, 3.1896707497480747e-13, 6.092903959142859e-13])
  end

  @testset "elixir_euler_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic_curved.jl"),
      l2   = [0.0018268326813744103, 0.001745601029521995, 0.001745601029521962, 0.0017456010295218891, 0.003239834454817457],
      linf = [0.014660503198892005, 0.01506958815284798, 0.01506958815283821, 0.015069588152864632, 0.02700205515651044])
  end

  @testset "elixir_euler_ec_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec_curved.jl"),
      l2   = [0.011367083018614027, 0.007022020327490176, 0.006759580335962235, 0.006820337637760632, 0.02912659127566544],
      linf = [0.2761764220925329, 0.20286331858055706, 0.18763944865434593, 0.19313636558790004, 0.707563913727584],
      tspan = (0.0, 0.25))
  end

  @testset "elixir_mhd_ec_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_curved.jl"),
      l2   = [0.009004021452004856, 0.007143957956917943, 0.00697375563632569, 0.0069016516181157755,
              0.03300636659954724, 0.0031996515755530713, 0.00307526880295712, 0.0030711742765310874,
              4.411700983795573e-5],
      linf = [0.29353751817533713, 0.2590666293828337, 0.26079069009197964, 0.24554085013509075,
              1.192029990255063, 0.12964835514332473, 0.1322601982698155, 0.14207453734982212,
              0.0036255415130575326],
      tspan = (0.0, 0.25))
  end

  @testset "elixir_mhd_alfven_wave_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_curved.jl"),
      l2   = [0.00695806796531185, 0.005905861769821461, 0.004679129482001242, 0.011372294079562632,
              0.008082731107429054, 0.009023698939340682, 0.007806531761609157, 0.011685969712999076,
              0.005069362185110317],
      linf = [0.13312729792145217, 0.07261792909268508, 0.056920016847370865, 0.19227757499888443,
              0.1288417336161367, 0.1491308063691154, 0.09147681280869269, 0.18478798034595412,
              0.07038490536912637],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_mhd_alfven_wave_curved.jl with flux_lax_friedrichs" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_curved.jl"),
      l2   = [0.0070749174389970765, 0.0059444827260872225, 0.00476655740426946, 0.011497279655843719,
              0.008132362559038092, 0.009120047939183473, 0.007889781639701823, 0.011820748188438215,
              0.005070060771140364],
      linf = [0.13393059418508102, 0.07680097207925154, 0.05922690804987149, 0.19837472391078007,
              0.13952706692370365, 0.15175230107829663, 0.09003746591104678, 0.19140045248967738,
              0.0724249033071008],
      tspan = (0.0, 0.5), surface_flux = flux_lax_friedrichs)
  end
end

end # module
