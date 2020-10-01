module TestExamples1D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

# Run basic tests
@testset "Examples 1D" begin
  @testset "parameters.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [5.581321238071356e-6],
            linf = [3.270561745361e-5])
  end
  @testset "parameters_amr.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_amr.toml"),
            l2   = [0.3540209654959832],
            linf = [0.9999905446337742])
  end
  @testset "parameters_blast_wave_shockcapturing.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_blast_wave_shockcapturing.toml"),
            l2   = [0.21530530948120738, 0.2805965425286348, 0.5591770920395336],
            linf = [1.508388610723991, 1.5622010377944118, 2.035149673163788],
            n_steps_max=30)
  end
  @testset "parameters_ec.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_ec.toml"),
            l2   = [0.11948926375393912, 0.15554606230413676, 0.4466895989733186],
            linf = [0.2956500342985863, 0.28341906267346123, 1.0655211913235232])
  end
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # Linear scalar advection
  @testset "parameters.toml with initial_conditions_sin" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [9.506162481381351e-5],
            linf = [0.00017492510098227054],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_sin")
  end
  @testset "parameters.toml with initial_conditions_constant" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [6.120436421866528e-16],
            linf = [1.3322676295501878e-15],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_constant")
  end
  @testset "parameters.toml with initial_conditions_linear_x" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters.toml"),
            l2   = [7.602419413667044e-17],
            linf = [2.220446049250313e-16],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_linear_x",
            boundary_conditions = "boundary_conditions_linear_x",
            periodicity=false)
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
