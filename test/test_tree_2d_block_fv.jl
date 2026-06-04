module TestTree2DBlockFV

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_block_FV")

@testset "BlockFV 2D" begin
#! format: noindent

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_block_FV.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_block_FV.jl"),
                        l2=[0.017295205942012868],
                        linf=[0.02444847499806624],
                        tspan=(0.0, 0.5))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Linear scalar advection

@testset "Compressible Euler equations" begin
#! format: noindent

@trixi_testset "elixir_euler_density_wave_block_FV.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_density_wave_block_FV.jl"),
                        l2=[0.5699714731043036, 0.05699714731043156,
                            0.11399429462086316, 0.014249286827611536],
                        linf=[0.8123668941820268, 0.08123668941820289,
                            0.16247337883640567, 0.020309172354558314],
                        tspan=(0.0, 0.5))

    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_euler_isentropic_vortex_block_FV.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_isentropic_vortex_block_FV.jl"),
                        l2=[0.001413036062339486, 0.07375960717268594,
                            0.07346841720884595, 0.1475370465736532],
                        linf=[0.022496747744819356, 0.7974470667978806,
                            0.7771847574190771, 2.333850189197271],
                        tspan=(0.0, 1.0))

    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_euler_convergence_block_FV.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_convergence_block_FV.jl"),
                        l2=[0.005308804975957337, 0.0066024481089594555,
                            0.006602448108959441, 0.028335881552453316],
                        linf=[0.00878493241051359, 0.010303479009843963,
                            0.010303479009844185, 0.04279338060505378],
                        tspan=(0.0, 0.5))

    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Compressible Euler equations
end # BlockFV 2D

end # module
