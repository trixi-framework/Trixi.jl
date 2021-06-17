
using BenchmarkTools
using Trixi

const SUITE = BenchmarkGroup()

let dimension = "2d"
  SUITE[dimension] = BenchmarkGroup()
  EXAMPLES_DIR = joinpath(examples_dir(), dimension)
  for elixir in [joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_advection_extended_curved.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic_curved.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_euler_ec_curved.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic_curved.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_wall_bc.jl"), # this is the only elixir working for polydeg=3
                 joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                 joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
                 joinpath(@__DIR__, "elixir_2d_euler_vortex_tree.jl"),
                 joinpath(@__DIR__, "elixir_2d_euler_vortex_structured.jl"),
                 joinpath(@__DIR__, "elixir_2d_euler_vortex_unstructured.jl")]
    SUITE[dimension][basename(elixir)] = BenchmarkGroup()
    for polydeg in [3, 7]
      trixi_include(elixir, tspan=(0.0, 1.0e-10); polydeg)
      SUITE[dimension][basename(elixir)]["p$(polydeg)_rhs!"] = @benchmarkable Trixi.rhs!(
        $(similar(sol.u[end])), $(copy(sol.u[end])), $(semi), $(first(tspan)))
      SUITE[dimension][basename(elixir)]["p$(polydeg)_analysis"] = @benchmarkable ($analysis_callback)($sol)
    end
  end
end

let dimension = "3d"
  SUITE[dimension] = BenchmarkGroup()
  EXAMPLES_DIR = joinpath(examples_dir(), dimension)
  for elixir in ["elixir_advection_extended.jl",
                 "elixir_advection_nonperiodic_curved.jl",
                 "elixir_euler_ec.jl",
                 "elixir_euler_ec_curved.jl",
                 "elixir_euler_nonperiodic_curved.jl",
                 "elixir_euler_mortar.jl",
                 "elixir_euler_shockcapturing.jl"]
    SUITE[dimension][basename(elixir)] = BenchmarkGroup()
    for polydeg in [3, 7]
      trixi_include(joinpath(EXAMPLES_DIR, elixir), tspan=(0.0, 1.0e-10); polydeg)
      SUITE[dimension][basename(elixir)]["p$(polydeg)_rhs!"] = @benchmarkable Trixi.rhs!(
        $(similar(sol.u[end])), $(copy(sol.u[end])), $(semi), $(first(tspan)))
      SUITE[dimension][basename(elixir)]["p$(polydeg)_analysis"] = @benchmarkable ($analysis_callback)($sol)
    end
  end
end

let
  SUITE["latency"] = BenchmarkGroup()
  SUITE["latency"]["default_example"] = @benchmarkable run(
    `$(Base.julia_cmd()) -e 'using Trixi; trixi_include(default_example())'`) seconds=60
  for polydeg in [3, 7]
    command = "using Trixi; trixi_include(joinpath(examples_dir(), \"2d\", \"elixir_advection_extended.jl\"), tspan=(0.0, 1.0e-10), polydeg=$(polydeg), save_restart=TrivialCallback(), save_solution=TrivialCallback())"
    SUITE["latency"]["polydeg_$polydeg"] = @benchmarkable run($`$(Base.julia_cmd()) -e $command`) seconds=60
  end
end
