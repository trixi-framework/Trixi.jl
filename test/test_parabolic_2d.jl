module TestExamplesParabolic2D

using Test
using Trixi

include("test_trixi.jl")


# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "SemidiscretizationHyperbolicParabolic (2D)" begin

  @trixi_testset "DGMulti 2D rhs_parabolic!" begin

    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    mesh = DGMultiMesh(dg, cells_per_dimension=(2, 2))

    # test with polynomial initial condition x^2 * y
    # test if we recover the exact second derivative
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0, equations)

    semi = SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic, initial_condition, dg)
    @test_nowarn_mod show(stdout, semi)
    @test_nowarn_mod show(stdout, MIME"text/plain"(), semi)
    @test_nowarn_mod show(stdout, boundary_condition_do_nothing)

    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # test "do nothing" BC just returns first argument
    @test boundary_condition_do_nothing(u0, nothing) == u0

    @unpack cache, cache_parabolic, equations_parabolic = semi
    @unpack gradients = cache_parabolic
    for dim in eachindex(gradients)
      fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    Trixi.calc_gradient!(gradients, ode.u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, cache, cache_parabolic)
    @unpack x, y, xq, yq = mesh.md
    @test getindex.(gradients[1], 1) ≈ 2 * xq .* yq
    @test getindex.(gradients[2], 1) ≈ xq.^2

    u_flux = similar.(gradients)
    Trixi.calc_viscous_fluxes!(u_flux, ode.u0, gradients, mesh, equations_parabolic,
                               dg, cache, cache_parabolic)
    @test u_flux[1] ≈ gradients[1]
    @test u_flux[2] ≈ gradients[2]

    du = similar(ode.u0)
    Trixi.calc_divergence!(du, ode.u0, t, u_flux, mesh, equations_parabolic, boundary_condition_periodic,
                           dg, semi.solver_parabolic, cache, cache_parabolic)
    @test getindex.(du, 1) ≈ 2 * y
  end

  @trixi_testset "DGMulti: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_advection_diffusion.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.2485803335154642],
      linf = [1.079606969242132]
    )
  end

  @trixi_testset "DGMulti: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_advection_diffusion_periodic.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.03180371984888462],
      linf = [0.2136821621370909]
    )
  end

  @trixi_testset "DGMulti: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_advection_diffusion_nonperiodic.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.002123168335604323],
      linf = [0.00963640423513712]
    )
  end

  @trixi_testset "DGMulti: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_navierstokes_convergence.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.0015355076812510957, 0.0033843168272696756, 0.0036531858107443434, 0.009948436427519214],
      linf = [0.005522560467190019, 0.013425258500730508, 0.013962115643482154, 0.027483102120502423]
    )
  end

  @trixi_testset "DGMulti: elixir_navierstokes_convergence_curved.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_navierstokes_convergence_curved.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.1),
      l2 = [0.004255101916146187, 0.011118488923215765, 0.011281831283462686, 0.03573656447388509],
      linf = [0.015071710669706473, 0.04103132025858458, 0.03990424085750277, 0.1309401718598764],
    )
  end

  @trixi_testset "DGMulti: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d", "elixir_navierstokes_lid_driven_cavity.jl"),
      cells_per_dimension = (4, 4), tspan=(0.0, 0.5),
      l2 = [0.00022156125227115747, 0.028318325921401, 0.009509168701070296, 0.028267900513550506],
      linf = [0.001562278941298234, 0.14886653390744856, 0.0716323565533752, 0.19472785105241996]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_diffusion.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.4), polydeg=5,
      l2 = [4.0915532997994255e-6],
      linf = [2.3040850347877395e-5]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_diffusion_nonperiodic.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.1),
      l2 = [0.007646800618485118],
      linf = [0.10067621050468958]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_convergence.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.1),
      l2 = [0.002111672530658797, 0.0034322351490857846, 0.0038742528195910416, 0.012469246082568561],
      linf = [0.012006418939223495, 0.035520871209746126, 0.024512747492231427, 0.11191122588756564]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (isothermal walls)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_convergence.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.1),
      heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x, t, equations), equations)),
      l2 = [0.002103629650383915, 0.003435843933396454, 0.00386735987813341, 0.012670355349235728],
      linf = [0.012006261793147788, 0.03550212518982032, 0.025107947319661185, 0.11647078036571124]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_convergence.jl"),
      initial_refinement_level=2, tspan=(0.0, 0.1), gradient_variables=GradientVariablesEntropy(),
      l2 = [0.0021403742517389513, 0.0034258287094908572, 0.0038915122886898517, 0.012506862343013842],
      linf = [0.012244412004628336, 0.03507559186162224, 0.024580892345558894, 0.11425600758350107]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables, isothermal walls)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_convergence.jl"),
      initial_refinement_level=2, tspan=(0.0, 0.1), gradient_variables=GradientVariablesEntropy(),
      heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x, t, equations), equations)),
      l2 = [0.0021349737347844907, 0.0034301388278203033, 0.0038928324474291572, 0.012693611436230873],
      linf = [0.01224423627586213, 0.035054066314102905, 0.025099598504931965, 0.11795616324751634]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (flux differencing)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_convergence.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.1),
      volume_integral=VolumeIntegralFluxDifferencing(flux_central),
      l2 = [0.0021116725306633594, 0.0034322351490827557, 0.0038742528196093542, 0.012469246082526909],
      linf = [0.012006418939291663, 0.035520871209594115, 0.024512747491801577, 0.11191122588591007]
    )
  end

  @trixi_testset "TreeMesh2D: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_lid_driven_cavity.jl"),
      initial_refinement_level = 2, tspan=(0.0, 0.5),
      l2 = [0.00015144571529699053, 0.018766076072331623, 0.007065070765652574, 0.0208399005734258],
      linf = [0.0014523369373669048, 0.12366779944955864, 0.05532450997115432, 0.16099927805328207]
    )
  end

end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
