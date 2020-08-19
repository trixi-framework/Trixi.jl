using Test
import Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)

# Numerical examples from the Euler-gravity paper
@testset "repro-self-gravitating-gas-dynamics" begin
  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_euler.toml" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_euler.toml",
            l2   = [0.00017409779099463607, 0.0003369287450282371, 0.00033692874502819616, 0.0006099035183426747],
            linf = [0.0010793454782482836, 0.0018836374478419238, 0.0018836374478410356, 0.003971446179607874])
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_euler.toml with N=4" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_euler.toml",
            l2   = [1.7187032983384504e-5, 2.6780178144541376e-5, 2.678017814469407e-5, 4.952410417693103e-5],
            linf = [0.00015018092862240096, 0.00016548331714294484, 0.00016548331714405506, 0.00043726245511699346],
            N=4)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_hyperbolic_diffusion.toml" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_hyperbolic_diffusion.toml",
            l2   = [0.00315402168051244, 0.012394424055283394, 0.021859728673870843],
            linf = [0.017332075119072865, 0.07843510773347322, 0.11325788389718668])
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_hyperbolic_diffusion.toml with N=4" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_hyperbolic_diffusion.toml",
            l2   = [0.00025112830138292663, 0.0008808243851096586, 0.0016313343234903468],
            linf = [0.001719090967553516, 0.0031291844657076145, 0.00994609342322228],
            N=4)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml",
            l2   = [0.0002487158370511598, 0.0003370291440916084, 0.00033702914409161063, 0.0007231934514459757, 0.00013852406160669235, 0.0007541252869723029, 0.0007541252869723299],
            linf = [0.001581173125044355, 0.002049389755695241, 0.0020493897556961294, 0.004793721268126383, 0.0009549587622960237, 0.0030981236291827237, 0.003098123629182964],
            t_end=0.1)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml with N=4" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml",
            l2   = [1.9536732064098693e-5, 2.756381055173374e-5, 2.7563810551703437e-5, 5.688705902953846e-5, 1.0684325470325204e-5, 5.829033623593028e-5, 5.829033623591347e-5],
            linf = [0.00012335977351507488, 0.00020086338378089152, 0.00020086338378044744, 0.0004962132679873221, 8.5358666522109e-5, 0.0002927883863423353, 0.00029278838634330673],
            t_end=0.1, N=4)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml with update_gravity_once_per_stage=false" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml",
            l2   = [0.00039754984075255105, 0.0011317710289437735, 0.00113177102894379, 0.002302567979915388, 0.0002449228820755184, 0.0009838245219995854, 0.0009838245219995676],
            linf = [0.0013753628419428399, 0.0031706120730756737, 0.0031706120730756737, 0.0069469754604232214, 0.0008251489152860739, 0.0030597255494218545, 0.0030597255494215977],
            t_end=0.1, update_gravity_once_per_stage=false)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml with 1st order RK3S*" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml",
            l2   = [0.00024871583705119436, 0.0003370291440915927, 0.0003370291440916112, 0.0007231934514459859, 0.00013852406160669225, 0.0007541252869723031, 0.0007541252869723208],
            linf = [0.001581173125044355, 0.002049389755695241, 0.0020493897556961294, 0.004793721268126383, 0.0009549587622960237, 0.0030981236291827237, 0.003098123629182964],
            t_end=0.1, time_integration_scheme_gravity="timestep_gravity_erk51_3Sstar!")
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml with 3rd order RK3S*" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_eoc_test_coupled_euler_gravity.toml",
            l2   = [0.000248715837047647, 0.0003370291440257414, 0.00033702914402587556, 0.0007231934513057375, 0.0001385240616293125, 0.0007541252869544295, 0.0007541252869544261],
            linf = [0.00158117312532835, 0.0020493897540796446, 0.0020493897540800887, 0.0047937212650124295, 0.000954958762033685, 0.0030981236303003834, 0.003098123630300921],
            t_end=0.1, time_integration_scheme_gravity="timestep_gravity_erk53_3Sstar!")
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_jeans_instability.toml" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_jeans_instability.toml",
            l2   = [10734.069674978307, 13357.14137049754, 7.430683770399547e-6, 26835.16654472354],
            linf = [15190.15931461379, 18890.86677707439, 3.42147931356359e-5, 37962.713187191635],
            t_end=0.1)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_jeans_instability.toml with update_gravity_once_per_stage=false" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_jeans_instability.toml",
            l2   = [10723.628704109766, 13336.232286560364, 7.886319075672209e-6, 26809.064110298692],
            linf = [15175.393292613328, 18861.296962089087, 5.736141054119072e-5, 37925.79784276709],
            t_end=0.1, update_gravity_once_per_stage=false)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_jeans_instability.toml with RK3S*" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_jeans_instability.toml",
            l2   = [10734.467064217813, 13358.172101425738, 7.590895434458968e-6, 26836.1600182602],
            linf = [15190.72004490532, 18892.323766605758, 3.585224859803839e-5, 37964.11502066627],
            t_end=0.1, time_integration_scheme_gravity="timestep_gravity_erk52_3Sstar!",
            cfl_gravity=1.2)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_sedov_self_gravity.toml" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_sedov_self_gravity.toml",
            l2   = [0.03611513647099618, 0.038354963134207776, 0.038354963134207706, 0.14517949232036176],
            linf = [2.0292827234185022, 2.1567998163498374, 2.156799816349837, 3.630512616468018],
            t_end=0.05)
  end

  @testset "../examples/repro-self-gravitating-gas-dynamics/parameters_sedov_self_gravity.toml with amr_interval=0" begin
    test_trixi_run("../examples/repro-self-gravitating-gas-dynamics/parameters_sedov_self_gravity.toml",
            l2   = [0.0015404291475135325, 0.006407595512985243, 0.006407595512985238, 0.011826209871361832],
            linf = [0.12012572662468202, 0.6026031599346305, 0.6026031599346306, 0.8675583800891971],
            t_end=0.005, amr_interval=0, initial_refinement_level=8)
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)
