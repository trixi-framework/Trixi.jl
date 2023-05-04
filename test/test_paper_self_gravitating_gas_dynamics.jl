module TestPaperSelfgravitatingGasDynamics

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "paper_self_gravitating_gas_dynamics")

# Numerical examples from the Euler-gravity paper
@testset "paper_self_gravitating_gas_dynamics" begin
  @trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
      l2   = [0.0001740977055972079, 0.0003369355182519592, 0.0003369355182518708, 0.0006099171220334989],
      linf = [0.001079347149189669, 0.0018836938381321389, 0.001883693838132583, 0.003971575376718217])
  end

  @trixi_testset "elixir_euler_convergence.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
      l2   = [1.7187201161597772e-5, 2.678065111772951e-5, 2.678065111783027e-5, 4.952504160091526e-5],
      linf = [0.0001501749544159381, 0.00016549482504535362, 0.00016549482504601976, 0.0004372960291432193],
      polydeg = 4)
  end


  @trixi_testset "elixir_hypdiff_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_convergence.jl"),
      l2   = [0.003154024896093942, 0.012394432074951856, 0.02185973823794725],
      linf = [0.01731850928579215, 0.07843510773347553, 0.11242300176349201])
  end

  @trixi_testset "elixir_hypdiff_convergence.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_convergence.jl"),
      l2   = [0.0002511283012128458, 0.0008808243846610255, 0.0016313343228567005],
      linf = [0.0017290715087938668, 0.003129184465704738, 0.01000728849316701],
      polydeg = 4)
  end


  @trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.00024871265138964204, 0.0003370077102132591, 0.0003370077102131964, 0.0007231525513793697],
      linf = [0.0015813032944647087, 0.0020494288423820173, 0.0020494288423824614, 0.004793821195083758],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_eulergravity_convergence.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [1.9537712148648045e-5, 2.7564396197947587e-5, 2.7564396197967635e-5, 5.688838772067586e-5],
      linf = [0.00012335710672761735, 0.00020086268350816283, 0.00020086268350727465, 0.0004962155455632278],
      tspan = (0.0, 0.1), polydeg = 4)
  end

  @trixi_testset "elixir_eulergravity_convergence.jl with 1st order RK3S*" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.00024871265138959434, 0.000337007710281087, 0.0003370077102811394, 0.0007231525515231289],
      linf = [0.0015813032941613958, 0.002049428843978518, 0.0020494288439798503, 0.004793821198143977],
      tspan = (0.0, 0.1), timestep_gravity=Trixi.timestep_gravity_erk51_3Sstar!)
  end

  @trixi_testset "elixir_eulergravity_convergence.jl with 3rd order RK3S*" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.0002487126513894034, 0.00033700771023049785, 0.00033700771023048245, 0.0007231525514158737],
      linf = [0.0015813032943847727, 0.002049428842844314, 0.0020494288428452023, 0.004793821195971937],
      tspan = (0.0, 0.1), timestep_gravity=Trixi.timestep_gravity_erk53_3Sstar!)
  end


  @trixi_testset "elixir_eulergravity_jeans_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_jeans_instability.jl"),
      l2   = [10733.63378538114, 13356.780607423452, 1.6722844879795038e-6, 26834.076821148774],
      linf = [15194.296424901113, 18881.481685044182, 6.809726988008751e-6, 37972.99700513482],
      tspan = (0.0, 0.1),
      atol = 4.0e-6 # the background field is reatively large, so this corresponds to our usual atol
      )
  end

  @trixi_testset "elixir_eulergravity_jeans_instability.jl with RK3S*" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_jeans_instability.jl"),
      l2   = [10734.598193238024, 13358.217234481384, 1.911011743371934e-6, 26836.487841241516],
      linf = [15195.661004798487, 18883.512035906537, 7.867948710816926e-6, 37976.408478975296],
      tspan = (0.0, 0.1),
      atol = 4.0e-6, # the background field is reatively large, so this corresponds to our usual atol
      parameters=ParametersEulerGravity(background_density=1.5e7,
                                        gravitational_constant=6.674e-8,
                                        cfl=2.4,
                                        resid_tol=1.0e-4,
                                        n_iterations_max=1000,
                                        timestep_gravity=timestep_gravity_erk52_3Sstar!))
  end

  @trixi_testset "Printing" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_jeans_instability.jl"),
      tspan = (0.0, 1.0e-5),
      parameters=ParametersEulerGravity(background_density=1.5e7,
                                        gravitational_constant=6.674e-8,
                                        cfl=2.4,
                                        resid_tol=1.0e-4,
                                        n_iterations_max=1000,
                                        timestep_gravity=timestep_gravity_erk52_3Sstar!))

    show(stdout, parameters)
    show(stdout, semi)
    show(stdout, semi_euler.boundary_conditions)
    show(stdout, TrivialCallback())
    show(stdout, equations_euler)
  end


  @trixi_testset "elixir_eulergravity_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_sedov_blast_wave.jl"),
      l2   = [0.046315994852653024, 0.0650818006233669, 0.06508180062336677, 0.4896707211656037],
      linf = [2.3874843337593776, 4.07876384374792, 4.07876384374792, 16.23914384809855],
      tspan = (0.0, 0.05),
      coverage_override = (maxiters=2,))
  end

  @trixi_testset "elixir_eulergravity_sedov_blast_wave.jl with ref-level=8 and no AMR" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_sedov_blast_wave.jl"),
      l2   = [0.00289222135995042, 0.013724813590853825, 0.013724813590853832, 0.05822904710548214],
      linf = [0.26361780693997594, 1.3908873830688688, 1.3908873830688688, 4.066701303607613],
      tspan = (0.0, 0.005), initial_refinement_level=8, amr_callback=TrivialCallback())
  end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive=true)

end #module
