using Trixi

trixi_include(default_example())
trixi_include(joinpath(examples_dir(), "2d", "elixir_euler_ec.jl"))
trixi_include(joinpath(examples_dir(), "2d", "elixir_euler_blast_wave_amr.jl"))
