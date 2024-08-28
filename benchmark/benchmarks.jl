# Disable formatting this file since it contains highly unusual formatting for better
# readability
#! format: off

using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using Trixi

const SUITE = BenchmarkGroup()

for elixir in [# 1D
               joinpath(examples_dir(), "structured_1d_dgsem", "elixir_euler_sedov.jl"),
               joinpath(examples_dir(), "tree_1d_dgsem", "elixir_mhd_ec.jl"), 
               joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence_walls_amr.jl"),
               joinpath(examples_dir(), "tree_1d_dgsem", "elixir_shallowwater_well_balanced_nonperiodic.jl"),
               # 2D
               joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_extended.jl"),
               joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr_nonperiodic.jl"),
               joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"),
               joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_vortex_mortar.jl"),
               joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_vortex_mortar_shockcapturing.jl"),
               joinpath(examples_dir(), "tree_2d_dgsem", "elixir_mhd_ec.jl"),
               joinpath(examples_dir(), "structured_2d_dgsem", "elixir_advection_extended.jl"),
               joinpath(examples_dir(), "structured_2d_dgsem", "elixir_advection_nonperiodic.jl"),
               joinpath(examples_dir(), "structured_2d_dgsem", "elixir_euler_ec.jl"),
               joinpath(examples_dir(), "structured_2d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
               joinpath(examples_dir(), "structured_2d_dgsem", "elixir_mhd_ec.jl"),
               joinpath(examples_dir(), "unstructured_2d_dgsem", "elixir_euler_wall_bc.jl"), # this is the only elixir working for polydeg=3
               joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_advection_extended.jl"),
               joinpath(@__DIR__, "elixir_2d_euler_vortex_tree.jl"),
               joinpath(@__DIR__, "elixir_2d_euler_vortex_structured.jl"),
               joinpath(@__DIR__, "elixir_2d_euler_vortex_unstructured.jl"),
               joinpath(@__DIR__, "elixir_2d_euler_vortex_p4est.jl"),
               # 3D
               joinpath(examples_dir(), "tree_3d_dgsem", "elixir_advection_extended.jl"),
               joinpath(examples_dir(), "tree_3d_dgsem", "elixir_euler_ec.jl"),
               joinpath(examples_dir(), "tree_3d_dgsem", "elixir_euler_mortar.jl"),
               joinpath(examples_dir(), "tree_3d_dgsem", "elixir_euler_shockcapturing.jl"),
               joinpath(examples_dir(), "tree_3d_dgsem", "elixir_mhd_ec.jl"),
               joinpath(examples_dir(), "structured_3d_dgsem", "elixir_advection_nonperiodic_curved.jl"),
               joinpath(examples_dir(), "structured_3d_dgsem", "elixir_euler_ec.jl"),
               joinpath(examples_dir(), "structured_3d_dgsem", "elixir_euler_source_terms_nonperiodic_curved.jl"),
               joinpath(examples_dir(), "structured_3d_dgsem", "elixir_mhd_ec.jl"),
               joinpath(examples_dir(), "p4est_3d_dgsem", "elixir_advection_basic.jl"),]
  benchname = basename(dirname(elixir)) * "/" * basename(elixir)
  SUITE[benchname] = BenchmarkGroup()
  for polydeg in [3, 7]
    trixi_include(elixir, tspan=(0.0, 1.0e-10); polydeg)
    SUITE[benchname]["p$(polydeg)_rhs!"] = @benchmarkable Trixi.rhs!(
      $(similar(sol.u[end])), $(copy(sol.u[end])), $(semi), $(first(tspan)))
    SUITE[benchname]["p$(polydeg)_analysis"] = @benchmarkable ($analysis_callback)($sol)
  end
end

let
  SUITE["latency"] = BenchmarkGroup()
  SUITE["latency"]["default_example"] = @benchmarkable run(
    `$(Base.julia_cmd()) --project=$(@__DIR__) -e 'using Trixi; trixi_include(default_example())'`) seconds=60
  for polydeg in [3, 7]
    command = "using Trixi; trixi_include(joinpath(examples_dir(), \"tree_2d_dgsem\", \"elixir_advection_extended.jl\"), tspan=(0.0, 1.0e-10), polydeg=$(polydeg), save_restart=TrivialCallback(), save_solution=TrivialCallback())"
    SUITE["latency"]["polydeg_$polydeg"] = @benchmarkable run(
      $`$(Base.julia_cmd()) --project=$(@__DIR__) -e $command`) seconds=60
  end
  SUITE["latency"]["euler_2d"] = @benchmarkable run(
    `$(Base.julia_cmd()) --project=$(@__DIR__) -e 'using Trixi; trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_kelvin_helmholtz_instability.jl"), tspan=(0.0, 1.0e-10), save_solution=TrivialCallback())'`) seconds=60
  SUITE["latency"]["mhd_2d"] = @benchmarkable run(
    `$(Base.julia_cmd()) --project=$(@__DIR__) -e 'using Trixi; trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_mhd_blast_wave.jl"), tspan=(0.0, 1.0e-10), save_solution=TrivialCallback())'`) seconds=60
end
