module TestAMDGPU3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "AMDGPU 3D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test real(ode.p.solver) == Float64
    @test real(ode.p.solver.basis) == Float64
    @test real(ode.p.solver.mortar) == Float64
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64
    @test eltype(ode.p.equations.advection_velocity) == Float64

    @test ode.u0 isa Array
    @test ode.p.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(ode.p.cache.elements) === Array
    @test Trixi.storage_type(ode.p.cache.interfaces) === Array
    @test Trixi.storage_type(ode.p.cache.boundaries) === Array
    @test Trixi.storage_type(ode.p.cache.mortars) === Array
end

@trixi_testset "elixir_advection_basic.jl Float32 / AMDGPU" begin
    # Using AMDGPU inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using AMDGPU
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(0.00016263963870641478)],
                        linf=[Float32(0.0014537194925779984)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=ROCArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1_700_000)
    @test real(ode.p.solver) == Float32
    @test real(ode.p.solver.basis) == Float32
    @test real(ode.p.solver.mortar) == Float32
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64
    @test eltype(ode.p.equations.advection_velocity) == Float32

    @test ode.u0 isa ROCArray
    @test ode.p.solver.basis.derivative_matrix isa ROCArray

    @test Trixi.storage_type(ode.p.cache.elements) === ROCArray
    @test Trixi.storage_type(ode.p.cache.interfaces) === ROCArray
    @test Trixi.storage_type(ode.p.cache.boundaries) === ROCArray
    @test Trixi.storage_type(ode.p.cache.mortars) === ROCArray
end

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[0.0014517629881062517,
                            0.0014469623017050836,
                            0.001446962301705153,
                            0.0014469623017051368,
                            0.002934065359862918],
                        linf=[0.01031578086475382,
                            0.011300883615913193,
                            0.011300883615896096,
                            0.011300883615918522,
                            0.02090696711453477],
                        volume_integral=VolumeIntegralFluxDifferencing(flux_kennedy_gruber))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test real(semi.solver) == Float64
    @test real(semi.solver.basis) == Float64
    @test real(semi.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float64

    @test ode.u0 isa Array
    @test semi.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(semi.cache.elements) === Array
    @test Trixi.storage_type(semi.cache.interfaces) === Array
    @test Trixi.storage_type(semi.cache.boundaries) === Array
    @test Trixi.storage_type(semi.cache.mortars) === Array
end

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl Float32 / AMDGPU" begin
    # Using AMDGPU inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using AMDGPU
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=Float32[0.0014518665391031068,
                                   0.0014470701356811022,
                                   0.0014470866449955344,
                                   0.00144707575575548,
                                   0.0029342928549885568],
                        linf=Float32[0.010317440030529479,
                                     0.011303550618318114,
                                     0.011295533976851013,
                                     0.011299068214785102,
                                     0.0209091211162149],
                        volume_integral=VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=ROCArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 600_000)
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa ROCArray
    @test semi.solver.basis.derivative_matrix isa ROCArray

    @test Trixi.storage_type(semi.cache.elements) === ROCArray
    @test Trixi.storage_type(semi.cache.interfaces) === ROCArray
    @test Trixi.storage_type(semi.cache.boundaries) === ROCArray
    @test Trixi.storage_type(semi.cache.mortars) === ROCArray
end

@trixi_testset "elixir_mhd_alfven_wave_nonperiodic.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonperiodic.jl"),
                        l2=[
                            0.00021050235921250785,
                            0.0006558863249658414,
                            0.0002821364462491609,
                            0.000794748439799794,
                            0.0006839039331448021,
                            0.0006743445567763623,
                            0.00031815692647892813,
                            0.0007885451813871558,
                            4.811726181476006e-5
                        ],
                        linf=[
                            0.0012031070458876636,
                            0.00410699976203599,
                            0.0017830978311310533,
                            0.004780625099412877,
                            0.0050959023689367555,
                            0.003922455896960386,
                            0.002515549812865392,
                            0.004448527707559019,
                            0.0001983994478820785
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test real(semi.solver) == Float64
    @test real(semi.solver.basis) == Float64
    @test real(semi.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float64

    @test ode.u0 isa Array
    @test semi.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(semi.cache.elements) === Array
    @test Trixi.storage_type(semi.cache.interfaces) === Array
    @test Trixi.storage_type(semi.cache.boundaries) === Array
    @test Trixi.storage_type(semi.cache.mortars) === Array
end

@trixi_testset "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32 / AMDGPU" begin
    # Using AMDGPU inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using AMDGPU
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=Float32[0.00021050235826592327, 0.0006558863204839041,
                                   0.0002821364444400733, 0.000794748435433683,
                                   0.0006839039307848098, 0.0006743445524692008,
                                   0.000318156924452865, 0.0007885451771559438,
                                   4.811726173404515e-5],
                        linf=Float32[0.0012031070350810857, 0.004106999758487398,
                                     0.001783097816025008, 0.004780625055122056,
                                     0.005095902318184908, 0.003922455893839549,
                                     0.002515549802432071, 0.004448527671538249,
                                     0.00019839944646198146],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=ROCArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 600_000)
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa ROCArray
    @test semi.solver.basis.derivative_matrix isa ROCArray

    @test Trixi.storage_type(semi.cache.elements) === ROCArray
    @test Trixi.storage_type(semi.cache.interfaces) === ROCArray
    @test Trixi.storage_type(semi.cache.boundaries) === ROCArray
    @test Trixi.storage_type(semi.cache.mortars) === ROCArray
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)
end
end # module
