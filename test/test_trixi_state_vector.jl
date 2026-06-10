module TestTrixiStateVector

using Test
using Trixi
using LinearAlgebra: norm, dot
using OrdinaryDiffEqCore: solve  # used in the Krylov integration test
using SciMLBase: ReturnCode

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TrixiStateVector" begin
#! format: noindent

# -------------------------------------------------------------------------
# Unit tests – AbstractVector interface, norm, dot, broadcast
# -------------------------------------------------------------------------

@timed_testset "AbstractVector interface" begin
    v = TrixiStateVector(Float64[1.0, 2.0, 3.0])
    @test size(v) == (3,)
    @test length(v) == 3
    @test eltype(v) == Float64
    @test v[2] == 2.0
    v[2] = 99.0
    @test v[2] == 99.0
    v[2] = 2.0

    # similar / copy / fill! / resize!
    s = similar(v)
    @test s isa TrixiStateVector{Float64}
    @test length(s) == 3

    s2 = similar(v, Float32)
    @test s2 isa TrixiStateVector{Float32}

    s3 = similar(v, Float32, (4,))
    @test s3 isa TrixiStateVector{Float32}
    @test length(s3) == 4

    c = copy(v)
    @test c isa TrixiStateVector
    @test c.data == v.data
    @test c.data !== v.data  # independent copy

    fill!(s, 7.0)
    @test all(s.data .== 7.0)

    copyto!(s, v)
    @test s.data == v.data

    resize!(s, 5)
    @test length(s) == 5
end

@timed_testset "Serial norm and dot" begin
    v = TrixiStateVector(Float64[3.0, 4.0])
    @test norm(v) ≈ 5.0           # sqrt(9+16)
    @test norm(v, 2) ≈ 5.0

    w = TrixiStateVector(Float64[1.0, 0.0])
    @test dot(v, w) ≈ 3.0         # 3*1 + 4*0

    # Verify norm(v)^2 == dot(v, v)
    @test norm(v)^2 ≈ dot(v, v)

    # Only p=2 is supported
    @test_throws ArgumentError norm(v, 1)
end

@timed_testset "Broadcast preserves TrixiStateVector wrapper" begin
    v = TrixiStateVector(Float64[1.0, 2.0, 3.0])
    k = TrixiStateVector(Float64[10.0, 20.0, 30.0])
    dt = 0.5

    # Out-of-place
    r = @. v + dt * k
    @test r isa TrixiStateVector
    @test r.data ≈ [6.0, 12.0, 18.0]

    # In-place stage update (typical ODE integrator pattern)
    u = TrixiStateVector(copy(v.data))
    @. u = u + dt * k
    @test u isa TrixiStateVector
    @test u.data ≈ [6.0, 12.0, 18.0]

    # Scalar broadcast
    @. u = 0.0
    @test all(u.data .== 0.0)
    @test u isa TrixiStateVector

    # Three-term linear combination
    a, b = 0.25, 0.75
    r2 = @. a * v + b * k
    @test r2 isa TrixiStateVector
    @test r2.data ≈ @. a * v.data + b * k.data
end

@timed_testset "ode_norm dispatches correctly for TrixiStateVector" begin
    v = TrixiStateVector(Float64[1.0, -2.0, 3.0])
    # ode_norm(::AbstractArray, t) normalises by length; verify it is callable and positive
    n = ode_norm(v, 0.0)
    @test n > 0
    # In serial: should equal sqrt(sum(abs2,v)/length(v))
    @test n ≈ sqrt(sum(abs2, v.data) / length(v.data))
end

# -------------------------------------------------------------------------
# Integration test – implicit time stepping with KrylovJL_GMRES
#
# TrixiStateVector overrides LinearAlgebra.dot and norm.  Krylov.jl (via
# LinearSolve.jl) calls these during GMRES iterations, so this test verifies
# end-to-end compatibility with matrix-free Newton-Krylov time integration.
# -------------------------------------------------------------------------

@timed_testset "KenCarp47 + KrylovJL_GMRES: TrixiStateVector matches plain Vector" begin
    using OrdinaryDiffEqSDIRK: KenCarp47
    using LinearSolve: KrylovJL_GMRES
    using ADTypes: AutoFiniteDiff

    # 1-D diffusion with LDG, same physics as
    # examples/tree_1d_dgsem/elixir_diffusion_ldg_newton_krylov.jl
    diffusivity_val = 0.5
    equations = LinearDiffusionEquation1D(diffusivity_val)
    solver = DGSEM(polydeg = 3)
    mesh = TreeMesh(-convert(Float64, pi), convert(Float64, pi),
                    initial_refinement_level = 4,
                    n_cells_max = 30_000, periodicity = true)

    function ic_diffusion(x, t, equation)
        nu = diffusivity_val
        return SVector(sin(sum(x)) * exp(-nu * t))
    end

    semi = SemidiscretizationParabolic(mesh, equations, ic_diffusion, solver;
                                       solver_parabolic = ParabolicFormulationLocalDG(),
                                       boundary_conditions = boundary_condition_periodic)
    ode = semidiscretize(semi, (0.0, 1.0); wrap_state = false)

    linsolve = KrylovJL_GMRES(atol = 1e-11, rtol = 1e-10)
    ode_alg = KenCarp47(autodiff = AutoFiniteDiff(), linsolve = linsolve)
    common_kwargs = (; abstol = 1e-10, reltol = 1e-9,
                     ode_default_options()...)

    # Reference: plain Vector (wrap_state=false)
    sol_plain = solve(ode, ode_alg; common_kwargs...)
    @test sol_plain.retcode == ReturnCode.Success

    # TrixiStateVector: same problem, u0 wrapped (wrap_state=true is the default)
    ode_tsv = semidiscretize(semi, (0.0, 1.0))
    sol_tsv = solve(ode_tsv, ode_alg; common_kwargs...)
    @test sol_tsv.retcode == ReturnCode.Success

    # Final states must agree to solver tolerance
    @test sol_tsv.u[end].data≈sol_plain.u[end] atol=1e-9
end
end # @testset "TrixiStateVector"

# Clean up afterwards: delete Trixi.jl output directory
isdir(outdir) && rm(outdir, recursive = true)

end # module
