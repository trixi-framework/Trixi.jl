#! format: off
using Trixi

trixi_include(default_example())
trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"))
trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_eulermulti_ec.jl"), tspan=(0.0, 0.1))
trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr_visualization.jl"))
trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_blast_wave_amr.jl"), tspan=(0.0, 1.0))
trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_euler_positivity.jl"), tspan=(0.0, 1.0))
trixi_include(joinpath(examples_dir(), "tree_3d_dgsem", "elixir_euler_shockcapturing_amr.jl"), tspan=(0.0, 0.1))
