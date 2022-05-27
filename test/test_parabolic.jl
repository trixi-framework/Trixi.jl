module TestExamplesDGMulti2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "SemidiscretizationHyperbolicParabolic" begin

  @trixi_testset "DGMulti 2D" begin

    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    mesh = DGMultiMesh(dg, cells_per_dimension=(2, 2))

    # test with polynomial initial condition x^2 * y
    # test if we recover the exact second derivative
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0)

    semi = SemidiscretizationHyperbolicParabolic(mesh, equation, equation_parabolic, initial_condition, dg)
    ode = semidiscretize(semi, (0.0, 0.01))

    @unpack cache, parabolic_cache, equations_parabolic = semi
    @unpack u_grad = parabolic_cache
    for dim in eachindex(u_grad)
      fill!(u_grad[dim], zero(eltype(u_grad[dim])))
    end

    Trixi.calc_gradient!(u_grad, ode.u0, mesh, equations_parabolic, boundary_condition_do_nothing, dg, cache, parabolic_cache)
    @unpack x, y = mesh.md
    @test getindex.(u_grad[1], 1) ≈ 2 * x .* y
    @test getindex.(u_grad[2], 1) ≈ x.^2

    u_flux = similar.(u_grad)
    Trixi.calc_viscous_fluxes!(u_flux, ode.u0, u_grad, mesh, equations_parabolic, dg, cache, parabolic_cache)
    @test u_flux[1] ≈ u_grad[1]
    @test u_flux[2] ≈ u_grad[2]

    du = similar(ode.u0)
    Trixi.calc_divergence!(du, ode.u0, u_flux, mesh, equations_parabolic, boundary_condition_do_nothing, dg, cache, parabolic_cache)
    @test getindex.(du, 1) ≈ 2 * y
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
