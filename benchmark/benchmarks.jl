
using BenchmarkTools
using Trixi

const SUITE = BenchmarkGroup()

let dimension = "2d"
  SUITE[dimension] = BenchmarkGroup()
  EXAMPLES_DIR = joinpath(examples_dir(), dimension)
  for elixir in ["elixir_advection_extended.jl", "elixir_advection_amr_nonperiodic.jl",
                  "elixir_euler_ec.jl", "elixir_euler_vortex_mortar.jl", "elixir_euler_vortex_mortar_shockcapturing.jl"]
    SUITE[dimension][elixir] = BenchmarkGroup()
    for polydeg in [3, 7]
      trixi_include(joinpath(EXAMPLES_DIR, elixir), tspan=(0.0, 0.0); polydeg)
      SUITE[dimension][elixir]["p$(polydeg)_rhs!"] = @benchmarkable Trixi.rhs!($(similar(ode.u0)), $(copy(ode.u0)), $(semi), $(first(tspan)))
      SUITE[dimension][elixir]["p$(polydeg)_analysis"] = @benchmarkable ($analysis_callback)($sol)
    end
  end
end

let dimension = "3d"
  SUITE[dimension] = BenchmarkGroup()
  EXAMPLES_DIR = joinpath(examples_dir(), dimension)
  for elixir in ["elixir_advection_extended.jl",
                 "elixir_euler_ec.jl", "elixir_euler_mortar.jl"]
    SUITE[dimension][elixir] = BenchmarkGroup()
    for polydeg in [3, 7]
      trixi_include(joinpath(EXAMPLES_DIR, elixir), tspan=(0.0, 0.0); polydeg)
      SUITE[dimension][elixir]["p$(polydeg)_rhs!"] = @benchmarkable Trixi.rhs!($(similar(ode.u0)), $(copy(ode.u0)), $(semi), $(first(tspan)))
      SUITE[dimension][elixir]["p$(polydeg)_analysis"] = @benchmarkable ($analysis_callback)($sol)
    end
  end
end

let
  SUITE["latency"] = BenchmarkGroup()
  SUITE["latency"]["default_example"] = @benchmarkable run(
    `$(Base.julia_cmd()) -e 'using Trixi; trixi_include(default_example())'`) seconds=60
  for polydeg in [3, 7]
    command = "using Trixi; trixi_include(joinpath(examples_dir(), \"2d\", \"elixir_advection_extended.jl\"), polydeg=$(polydeg), save_restart=TrivialCallback(), save_solution=TrivialCallback(), cfl=0.1)"
    SUITE["latency"]["polydeg_$polydeg"] = @benchmarkable run($`$(Base.julia_cmd()) -e $command`) seconds=60
  end
end
