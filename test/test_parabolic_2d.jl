@testsnippet Parabolic2D begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "Parabolic2D: DGMulti 2D rhs_parabolic!" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    using Trixi
    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    cells_per_dimension = (2, 2)
    mesh = DGMultiMesh(dg, cells_per_dimension)

    # test with polynomial initial condition x^2 * y
    # test if we recover the exact second derivative
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0, equations)

    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg;
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
    @trixi_test_nowarn show(stdout, semi)
    @trixi_test_nowarn show(stdout, MIME"text/plain"(), semi)
    @trixi_test_nowarn show(stdout, boundary_condition_do_nothing)

    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # test "do nothing" BC just returns first argument
    @test boundary_condition_do_nothing(u0, nothing) == u0

    (; cache, cache_parabolic, equations_parabolic) = semi
    (; gradients) = cache_parabolic
    for dim in eachindex(gradients)
        fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    # unpack VectorOfArray
    u0 = Base.parent(ode.u0)
    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    parabolic_scheme = semi.solver_parabolic
    Trixi.calc_gradient!(gradients, u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, parabolic_scheme,
                         cache, cache_parabolic)
    (; x, y, xq, yq) = mesh.md
    @test getindex.(gradients[1], 1) ≈ 2 * xq .* yq
    @test getindex.(gradients[2], 1) ≈ xq .^ 2

    u_flux = similar.(gradients)
    Trixi.calc_parabolic_fluxes!(u_flux, u0, gradients, mesh,
                                 equations_parabolic,
                                 dg, cache, cache_parabolic)
    @test u_flux[1] ≈ gradients[1]
    @test u_flux[2] ≈ gradients[2]

    du = similar(u0)
    Trixi.calc_divergence!(du, u0, t, u_flux, mesh,
                           equations_parabolic,
                           boundary_condition_periodic,
                           dg, semi.solver_parabolic, cache, cache_parabolic)
    Trixi.invert_jacobian!(du, mesh, equations_parabolic, dg, cache; scaling = 1.0)
    @test getindex.(du, 1) ≈ 2 * y
end

@testitem "Parabolic2D: DGMulti 2D rhs_parabolic! (Neumann BCs, source terms)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    using Trixi
    left(x, tol = 50 * eps()) = abs(x[1] + 1) < tol
    right(x, tol = 50 * eps()) = abs(x[1] - 1) < tol
    bottom(x, tol = 50 * eps()) = abs(x[2] + 1) < tol
    top(x, tol = 50 * eps()) = abs(x[2] - 1) < tol
    is_on_boundary = (; left, right, top, bottom)

    diffusivity = 0.7
    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(diffusivity, equations)

    # u = (x^2 + y^2) / 2 has the outward normal derivative 1 on all boundaries
    # of [-1, 1]^2 and the Laplacian 2.
    initial_condition = (x, t, equations) -> SVector(0.5f0 * (x[1]^2 + x[2]^2))
    boundary_condition = BoundaryConditionNeumann((x, t, equations) -> SVector(diffusivity))
    boundary_conditions_parabolic = (; left = boundary_condition,
                                     right = boundary_condition,
                                     top = boundary_condition,
                                     bottom = boundary_condition)
    # polynomial of degree 3 that is projected exactly
    source_terms_parabolic = (u, gradients, x, t, equations) -> SVector(x[1] * x[2]^2 +
                                                                        u[1] *
                                                                        gradients[1][1])

    for (element_type, approximation_type) in ((Tri(), Polynomial()), (Quad(), SBP()))
        dg = DGMulti(; polydeg = 3, element_type, approximation_type,
                     surface_integral = SurfaceIntegralWeakForm(flux_central),
                     volume_integral = VolumeIntegralWeakForm())
        mesh = DGMultiMesh(dg, (2, 2); is_on_boundary)
        (; x, y, xq, yq) = mesh.md

        for source_terms in (nothing, source_terms_parabolic)
            semi = SemidiscretizationHyperbolicParabolic(mesh,
                                                         (equations, equations_parabolic),
                                                         initial_condition, dg;
                                                         source_terms_parabolic = source_terms,
                                                         boundary_conditions = (boundary_condition_do_nothing,
                                                                                boundary_conditions_parabolic))
            u_ode = compute_coefficients(0.0, semi)
            du_ode = similar(u_ode)
            Trixi.rhs_parabolic!(du_ode, u_ode, semi, 0.0)
            du = Trixi.wrap_array(du_ode, semi)

            (; gradients) = semi.cache_parabolic
            @test getindex.(gradients[1], 1) ≈ xq
            @test getindex.(gradients[2], 1) ≈ yq

            du_exact = fill(2 * diffusivity, size(x))
            if !isnothing(source_terms)
                @. du_exact += x * y^2 + 0.5 * (x^2 + y^2) * x
            end
            @test getindex.(du, 1) ≈ du_exact
        end
    end
end

@testitem "Parabolic2D: DGMulti: elixir_advection_diffusion.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.2485803335154642],
                        linf=[1.079606969242132])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_advection_diffusion_periodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion_periodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.03180371984888462],
                        linf=[0.2136821621370909])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_advection_diffusion_nonperiodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.002123168335604323],
                        linf=[0.00963640423513712])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_convergence.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0015355076237431118,
                            0.003384316785885901,
                            0.0036531858026850757,
                            0.009948436101649498
                        ],
                        linf=[
                            0.005522560543588462,
                            0.013425258431728926,
                            0.013962115936715924,
                            0.027483099961148838
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_navierstokes_convergence_curved.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_convergence_curved.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0042551020940351444,
                            0.011118489080358264,
                            0.011281831362358863,
                            0.035736565778376306
                        ],
                        linf=[
                            0.015071709836357083,
                            0.04103131887989486,
                            0.03990424032494211,
                            0.13094018584692968
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: DGMulti: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.5),
                        l2=[
                            0.0002215612357465129,
                            0.028318325887331217,
                            0.009509168805093485,
                            0.028267893004691534
                        ],
                        linf=[
                            0.0015622793960574644,
                            0.1488665309341318,
                            0.07163235778907852,
                            0.19472797949052278
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[4.0915532997994255e-6],
                        linf=[2.3040850347877395e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        solver=DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs,
                                     basis_type = GaussLegendreBasis),
                        initial_refinement_level=2, tspan=(0.0, 0.4),
                        l2=[2.8254621369070895e-6], linf=[6.914648264633172e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[6.193056910594806e-6], linf=[4.918855889635143e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=3,
                        l2=[0.0015151641634070158], linf=[0.00956001376594895])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_gradient_source_terms.jl (Fixed timestep)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4),
                        solver_parabolic=ParabolicFormulationBassiRebay1(), nu=1e-3,
                        stepsize_callback=TrivialCallback(), dt=1e-1,
                        l2=[0.0017395186758592685], linf=[0.007481527467476025])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        mesh=P4estMesh((4, 4), polydeg = 1,
                                       coordinates_min = coordinates_min,
                                       coordinates_max = coordinates_max,
                                       periodicity = true),
                        tspan=(0.0, 0.4),
                        solver_parabolic=ParabolicFormulationBassiRebay1(), nu=1e-3,
                        stepsize_callback=TrivialCallback(), dt=1e-1,
                        l2=[0.0017395186758592685], linf=[0.007481527467476025])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion.jl (Refined mesh)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0))
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 8 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 8)])
    tspan = (0.0, 1.5)
    semi = SemidiscretizationHyperbolicParabolic(mesh,
                                                 (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic))
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [1.67452550744728e-6]
    @test linf_error ≈ [7.905059166368744e-6]

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 100)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 100)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_amr.jl"),
                        initial_refinement_level=2,
                        base_level=2,
                        med_level=3,
                        max_level=4,
                        l2=[0.0009662045510830027],
                        linf=[0.006121646998993091])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[0.007646800618485118],
                        linf=[0.10067621050468958])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                                     basis_type = GaussLegendreBasis),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[0.005916696880764326],
                        linf=[0.034212013224034776])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.0007711488519400885],
                        linf=[0.015254743335726637])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.01),
                        l2=[0.000684755734524055],
                        linf=[0.01141444199847298],
                        # The AMR indicator and the adaptive time stepping react
                        # sensitively to roundoff differences between environments and
                        # package versions, so relax the error tolerances.
                        rtol=1e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_imex_operator.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_imex_operator.jl"),
                        l2=[3.869738113458773e-6], linf=[1.9950342183605585e-5],
                        # The implicit solution depends on the environment and the
                        # package versions at the 1e-6 level, so relax the error
                        # tolerances.
                        rtol=1e-5,
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    # Compile Trixi.rhs_parabolic! by calling it once before checking
    # allocations
    Trixi.rhs_parabolic!(similar(sol.u[end]), copy(sol.u[end]), semi, sol.t[end])
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[0.007009146246373517], linf=[0.09535203925012649])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           extra_analysis_integrals = (energy_kinetic,
                                                                                       energy_internal,
                                                                                       enstrophy)),
                        l2=[
                            0.0021116725306624543,
                            0.003432235149083229,
                            0.003874252819605527,
                            0.012469246082535005
                        ],
                        linf=[
                            0.012006418939279007,
                            0.03552087120962882,
                            0.02451274749189282,
                            0.11191122588626357
                        ])
    reference_solution = copy(sol.u[end])

    # The specific gas constant only changes the conversion from pressure to
    # temperature. Thus, the solution in conserved variables must be independent
    # of R (unless a temperature-dependent viscosity is used, e.g., Sutherland's law).
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                R = 2.0))
    @test sol.u[end] ≈ reference_solution

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (isothermal walls)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    using Trixi: Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021036296503840883,
                            0.003435843933397192,
                            0.003867359878114748,
                            0.012670355349293195
                        ],
                        linf=[
                            0.01200626179308184,
                            0.03550212518997239,
                            0.025107947320178275,
                            0.11647078036751068
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.002140374251729127,
                            0.003425828709496601,
                            0.0038915122887358097,
                            0.012506862342858291
                        ],
                        linf=[
                            0.012244412004772665,
                            0.03507559186131113,
                            0.02458089234472249,
                            0.11425600758024679
                        ])
    reference_solution = copy(sol.u[end])

    # The specific gas constant only changes the conversion from pressure to
    # temperature. Thus, the solution in conserved variables must be independent
    # of R (unless a temperature-dependent viscosity is used, e.g., Sutherland's law).
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                R = 2.0,
                                                                                gradient_variables = GradientVariablesEntropy()))
    @test sol.u[end] ≈ reference_solution

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables, isothermal walls)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021349737347923716,
                            0.0034301388278178365,
                            0.0038928324473968836,
                            0.012693611436338
                        ],
                        linf=[
                            0.012244236275761766,
                            0.03505406631430898,
                            0.025099598505644406,
                            0.11795616324985403
                        ])
    reference_solution = copy(sol.u[end])

    # At an isothermal wall, the last entropy variable is -1 / (R * T).
    # Therefore, changing R changes the prescribed temperature but must not
    # change the solution in conserved variables.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                R = 6.7,
                                                                                gradient_variables = GradientVariablesEntropy()),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)))
    @test sol.u[end] ≈ reference_solution

    # A temperature-dependent viscosity following Sutherland's law with reference
    # temperature `T_ref` and Sutherland constant `C`. It is invariant under scaling
    # `T`, `T_ref`, and `C` by the same factor.
    function mu_sutherland(T_ref, C)
        return function (w, equations)
            T = last(Trixi.entropy2velocity_temperature(w, equations))
            return 0.01 * (T_ref + C) / (T + C) * (T / T_ref)^1.5
        end
    end

    # With R = 1, the temperature T = p / rho = rho of the initial condition is in [1.5, 2.5].
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu_sutherland(2.0,
                                                                                                   1.0),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)))
    reference_solution_sutherland = copy(sol.u[end])
    # The temperature dependence of the viscosity changes the solution
    @test !isapprox(reference_solution_sutherland, reference_solution, rtol = 1.0e-6)

    # Changing R scales the temperature by 1 / R. Thus, scaling the parameters of
    # Sutherland's law accordingly must not change the solution.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu_sutherland(2.0 /
                                                                                                   6.7,
                                                                                                   1.0 /
                                                                                                   6.7),
                                                                                Prandtl = prandtl_number(),
                                                                                R = 6.7,
                                                                                gradient_variables = GradientVariablesEntropy()),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)))
    @test sol.u[end] ≈ reference_solution_sutherland

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (flux differencing)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_central),
                        l2=[
                            0.0021116725306612075,
                            0.0034322351490838703,
                            0.0038742528196011594,
                            0.012469246082545557
                        ],
                        linf=[
                            0.012006418939262131,
                            0.0355208712096602,
                            0.024512747491999436,
                            0.11191122588669522
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_convergence.jl (Refined mesh)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        tspan=(0.0, 0.0), initial_refinement_level=3)
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 4 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 4)])
    tspan = (0.0, 0.5)
    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic),
                                                 source_terms = source_terms_navier_stokes_convergence_test)
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                dt = 1e-5,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈
          [0.00024296959174050973;
           0.00020932631586399853;
           0.0005390572390981241;
           0.00026753561391316933]
    @test linf_error ≈
          [0.0016210102053486608;
           0.0025932876486537016;
           0.0029539073438284817;
           0.0020771191202548778]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=[
                            0.00015144571529699053,
                            0.018766076072331623,
                            0.007065070765652574,
                            0.0208399005734258
                        ],
                        linf=[
                            0.0014523369373669048,
                            0.12366779944955864,
                            0.05532450997115432,
                            0.16099927805328207
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer.jl"),
                        l2=[
                            0.0003588629975695495,
                            0.09679169095152979,
                            0.02016350767451026,
                            0.08926942734274035
                        ],
                        linf=[
                            0.001389809911103601,
                            0.7556236253978281,
                            0.06288444501194647,
                            0.36751558085282454
                        ],
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer.jl (R = 2)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer.jl"),
                        R=2.0, T_ref=40.0,
                        l2=[
                            0.0003588629975695495,
                            0.09679169095152979,
                            0.02016350767451026,
                            0.08926942734274035
                        ],
                        linf=[
                            0.001389809911103601,
                            0.7556236253978281,
                            0.06288444501194647,
                            0.36751558085282454
                        ],
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_amr.jl"),
                        l2=[
                            0.0003791330378826807,
                            0.09969310227471417,
                            0.021308426486134697,
                            0.09122052535455805
                        ],
                        linf=[
                            0.001566286906820924,
                            0.6899677733045734,
                            0.0643250936249398,
                            0.5512608650028881
                        ],
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        l2=[
                            0.0008995637356007158,
                            0.12576755986841737,
                            0.029891283547955904,
                            0.21166343892606548
                        ],
                        linf=[
                            0.0026654558931118455,
                            0.9086814268850737,
                            0.09591868454130359,
                            0.6827486530181943
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[
                            0.000899495992179304,
                            0.12562789588606163,
                            0.029899931669826938,
                            0.2116954741587301
                        ],
                        linf=[
                            0.0027386220646745674,
                            0.9169147177539221,
                            0.09573487541641176,
                            0.7175285501667474
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_taylor_green_vortex_sutherland.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex_sutherland.jl"),
                        time_int_tol=1e-9,
                        l2=[
                            0.0014455667914214441,
                            0.0010217565204670381,
                            0.001021756520467803,
                            0.011214256778300569
                        ],
                        linf=[
                            0.0032703168941358163,
                            0.003948810274891024,
                            0.0039488102748954645,
                            0.0250517046202674
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_viscous_shock.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            2.817640352994614e-5,
                            1.3827802177935697e-5,
                            3.1001993851549174e-17,
                            1.753568238029073e-5
                        ],
                        linf=[
                            0.0002185837290411552,
                            0.00013405261969601234,
                            1.8273738729889617e-16,
                            0.00015782934605046428
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_navierstokes_viscous_shock.jl (Gauss-Legendre, LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_hlle,
                                     basis_type = GaussLegendreBasis),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[
                            6.599006356353962e-6,
                            4.51480519387225e-6,
                            6.589047152614131e-17,
                            4.882545505551843e-6
                        ],
                        linf=[
                            3.7580718257101964e-5,
                            2.6691756677132972e-5,
                            4.978990988213157e-16,
                            2.9894348934300652e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_periodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.0023754695605828443],
                        linf=[0.008154128363741964])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_rotated.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_rotated.jl"),
                        l2=[1.9587359234701733e-5], linf=[0.000577240851833416])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_periodic_curved.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.006708147442490916],
                        linf=[0.04807038397976693])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_periodic_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[4.49813737871379e-5],
                        linf=[0.0001874702347290924])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.0007711488519390165],
                        linf=[0.015254743335703765])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl (Parabolic CFL)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        initial_refinement_level=2,
                        callbacks=CallbackSet(summary_callback, analysis_callback,
                                              alive_callback,
                                              StepsizeCallback(cfl = 1.6,
                                                               cfl_parabolic = 0.2)),
                        ode_alg=CarpenterKennedy2N54(williamson_condition = false),
                        dt=1.0, # will be overwritten
                        l2=[0.00010850375815619432],
                        linf=[0.0024081141187764932])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.01),
                        l2=[0.0006847533999311489],
                        linf=[0.01141430509080712],
                        # The AMR indicator and the adaptive time stepping react
                        # sensitively to roundoff differences between environments and
                        # package versions, so relax the error tolerances.
                        rtol=1e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: nonconforming meshes (linear solution)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    # The parabolic terms must vanish for a linear solution, in particular
    # at nonconforming interfaces (mortars). The mortars are created by the
    # initial adaptation of the AMR callbacks.
    initial_condition_linear(x, t, equations) = SVector(1 + x[1] + 2 * x[2])
    # Smaller values to match the thresholds of the AMR controller in
    # `elixir_advection_diffusion_amr_inverted_index.jl`
    initial_condition_linear_small(x, t, equations) = SVector(0.05 * (x[1] + x[2]))

    function max_abs_rhs_parabolic(semi)
        # Use the mesh after the initial adaptation of the AMR callback
        u_ode = Trixi.compute_coefficients(0.0, semi)
        du_ode = similar(u_ode)
        Trixi.rhs_parabolic!(du_ode, u_ode, semi, 0.0)
        return maximum(abs, du_ode)
    end

    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        initial_condition=initial_condition_linear,
                        tspan=(0.0, 0.0))
    @test Trixi.nmortars(solver, semi.cache) > 0
    @test max_abs_rhs_parabolic(semi) < 1.0e-9

    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        initial_condition=initial_condition_linear,
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.0))
    @test Trixi.nmortars(solver, semi.cache) > 0
    @test max_abs_rhs_parabolic(semi) < 1.0e-9

    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        initial_condition=initial_condition_linear,
                        tspan=(0.0, 0.0))
    @test Trixi.nmortars(solver, semi.cache) > 0
    @test max_abs_rhs_parabolic(semi) < 1.0e-9

    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        initial_condition=initial_condition_linear,
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.0))
    @test Trixi.nmortars(solver, semi.cache) > 0
    @test max_abs_rhs_parabolic(semi) < 1.0e-9

    # Unstructured curved mesh with mortars between elements with different orientations
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_amr_inverted_index.jl"),
                        initial_condition=initial_condition_linear_small,
                        boundary_condition=BoundaryConditionDirichlet(initial_condition),
                        tspan=(0.0, 0.0))
    @test Trixi.nmortars(solver, semi.cache) > 0
    @test max_abs_rhs_parabolic(semi) < 1.0e-9

    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_amr_inverted_index.jl"),
                        initial_condition=initial_condition_linear_small,
                        boundary_condition=BoundaryConditionDirichlet(initial_condition),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        tspan=(0.0, 0.0))
    @test Trixi.nmortars(solver, semi.cache) > 0
    @test max_abs_rhs_parabolic(semi) < 1.0e-9
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_curved.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.00919917034843865],
                        linf=[0.14186297438393505])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_advection_diffusion_nonperiodic_curved.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    # Serial reference for the corresponding MPI test in `test_mpi_p4est_parabolic_2d.jl`
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_curved.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.00912383895722383],
                        linf=[0.14071539996402677])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0003811978986531135,
                            0.0005874314969137914,
                            0.0009142898787681551,
                            0.0011613918893790497
                        ],
                        linf=[
                            0.0021633623985426453,
                            0.009484348273965089,
                            0.0042315720663082534,
                            0.011661660264076446
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_convergence.jl (Float32)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    # Only set up the semidiscretization, the time integration is done below
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                           "elixir_navierstokes_convergence.jl"),
                  initial_refinement_level = 1, tspan = (0.0, 0.01), sol = nothing)
    semi32 = trixi_adapt(semi; real_type = Float32, storage_type = Array)
    ode = semidiscretize(semi32, tspan)
    @test real(semi32.solver) == Float32
    @test typeof(semi32.equations_parabolic.mu) == Float32
    @test ode.u0 isa Vector{Float32}
    @test eltype(ode.tspan) == Float32
    parabolic_container = semi32.cache_parabolic.parabolic_container
    @test parabolic_container.u_transformed isa Array{Float32, 4}
    @test all(x -> x isa Array{Float32, 4}, parabolic_container.gradients)
    @test all(x -> x isa Array{Float32, 4}, parabolic_container.flux_parabolic)

    # The right-hand side agrees with the one of the original `Float64` semidiscretization
    t = 0.005
    u_ode = compute_coefficients(t, semi)
    du_ode = similar(u_ode)
    u_ode32 = Float32.(u_ode)
    du_ode32 = similar(u_ode32)
    Trixi.rhs_parabolic!(du_ode, u_ode, semi, t)
    Trixi.rhs_parabolic!(du_ode32, u_ode32, semi32, Float32(t))
    @test du_ode32 ≈ du_ode
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, t)
    Trixi.rhs_hyperbolic!(du_ode32, u_ode32, semi32, Float32(t))
    @test du_ode32 ≈ du_ode

    # A few time steps in `Float32` stay close to the `Float64` solution
    ode64 = semidiscretize(semi, (0.0, 0.01))
    alg = CarpenterKennedy2N54(williamson_condition = false)
    sol32 = solve(ode, alg; dt = 1.0f-3, adaptive = false, save_everystep = false)
    sol64 = solve(ode64, alg; dt = 1.0e-3, adaptive = false, save_everystep = false)
    @test sol32.u[end] isa Vector{Float32}
    @test sol32.u[end] ≈ sol64.u[end]
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_convergence_nonperiodic.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence_nonperiodic.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0004036496258545996,
                            0.0005869762480189079,
                            0.0009148853742181908,
                            0.0011984191532764543
                        ],
                        linf=[
                            0.0024993634989209923,
                            0.009487866203496731,
                            0.004505829506103787,
                            0.011634902753554499
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=[
                            0.00028716166408816073,
                            0.08101204560401647,
                            0.02099595625377768,
                            0.05008149754143295
                        ],
                        linf=[
                            0.014804500261322406,
                            0.9513271652357098,
                            0.7223919625994717,
                            1.4846907331004786
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity.jl (R=2)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu,
                                                                                Prandtl = prandtl_number(),
                                                                                R = 2.0),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        # Exact same errors as for the R = 1 case above
                        l2=[
                            0.00028716166408816073,
                            0.08101204560401647,
                            0.02099595625377768,
                            0.05008149754143295
                        ],
                        linf=[
                            0.014804500261322406,
                            0.9513271652357098,
                            0.7223919625994717,
                            1.4846907331004786
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity_amr.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity_amr.jl"),
                        tspan=(0.0, 1.0),
                        l2=[
                            0.0005305062668392406, 0.07898766423138423,
                            0.02928291476915143, 0.11715767702870489
                        ],
                        linf=[
                            0.005950675379011616, 0.9254051903170974,
                            0.7991033306362262, 1.694659748457326
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_lid_driven_cavity_amr.jl (IndicatorNodalFunction)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity_amr.jl"),
                        # Stop before t ≈ 1.1: afterwards, the mesh is very coarse and
                        # round-off differences (e.g., from different CPUs) are amplified
                        # by the adaptive time stepping to differences of ~1e-4 in the
                        # errors.
                        tspan=(0.0, 0.3),
                        amr_indicator=IndicatorNodalFunction((u, x, t) -> ((x[1] <
                                                                            sin(π * t)) &&
                                                                           (x[2] <
                                                                            sin(π * t))) ?
                                                                          1.0 : 0.0,
                                                             semi),
                        l2=[
                            0.0003261584331979379,
                            0.052803388370324895,
                            0.012246174341480693,
                            0.06238078545560245
                        ],
                        linf=[
                            0.004327085006800124,
                            0.5507651353359829,
                            0.20814988531441148,
                            0.5983357262773836
                        ])
    # Ensure that the mesh was coarsened and refined as expected to test
    # IndicatorNodalFunction: N_ele(t) = 576 (initial), 336 (t ≈ 0.077),
    # 354 (t ≈ 0.15), 444 (t ≈ 0.24)
    @test nelements(semi.cache.elements) == 444
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        l2=[
                            0.0008995637356007025,
                            0.1257675598683805,
                            0.02989128354793613,
                            0.21166343892604778
                        ],
                        linf=[
                            0.002665455893114288,
                            0.9086814268862065,
                            0.09591868454131727,
                            0.682748653018308
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_shearlayer_nonconforming.jl (LDG)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_nonconforming.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        l2=[
                            0.0008994959921791527,
                            0.125627895886022,
                            0.029899931669806482,
                            0.21169547415869083
                        ],
                        linf=[
                            0.0027386220646294923,
                            0.9169147177513159,
                            0.09573487541498969,
                            0.7175285501578514
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_NACA0012airfoil_mach08.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                        l2=[0.000186486564226516,
                            0.0005076712323400374,
                            0.00038074588984354107,
                            0.002128177239782089],
                        linf=[0.5153387072802718,
                            1.199362305026636,
                            0.9077214424040279,
                            5.666071182328691], tspan=(0.0, 0.001),
                        initial_refinement_level=0)

    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode) # Just a placeholder in this case

    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)

    drag_p = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                           semi.cache, semi)
    lift_p = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                           semi.cache, semi)

    drag_f = Trixi.analyze(drag_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)
    lift_f = Trixi.analyze(lift_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)

    @test isapprox(drag_p, 0.17963843913309516, atol = 1e-13)
    @test isapprox(lift_p, 0.26462588007949367, atol = 1e-13)

    @test isapprox(drag_f, 1.5427441885921553, atol = 1e-13)
    @test isapprox(lift_f, 0.005621910087395724, atol = 1e-13)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # We move these tests here to avoid modifying values used
    # to compute the drag/lift coefficients above.
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_NACA0012airfoil_mach08.jl (entropy gradient variables)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion2D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[0.00018594980973881656,
                            0.0005125796180749062,
                            0.00038371304694500115,
                            0.0021157711045400283],
                        linf=[0.5125645624266784,
                            1.2347607775786207,
                            0.9219577223090923,
                            5.609128002255419], tspan=(0.0, 0.001),
                        initial_refinement_level=0)

    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode) # Just a placeholder in this case

    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)

    # The shear stress is computed from the gradients of the entropy variables.
    # The values are close to the ones obtained with `GradientVariablesPrimitive`
    # above (drag coefficient 1.5427441885921553).
    drag_f = Trixi.analyze(drag_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)
    lift_f = Trixi.analyze(lift_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)

    @test isapprox(drag_f, 1.5544983191877073, atol = 1e-13)
    @test isapprox(lift_f, 0.010271534458877359, atol = 1e-13)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_NACA0012airfoil_mach085_restart.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach085_restart.jl"),
                        l2=[
                            3.257684235589157e-6,
                            4.7338328545109256e-5,
                            6.518920877620882e-5,
                            0.003637221820407737
                        ],
                        linf=[
                            0.0020404564102322843,
                            0.05933631737841297,
                            0.03667271766233409,
                            2.033602537228147
                        ],
                        tspan=(0.0, 0.002),
                        # number of time steps of the base elixir up to `tspan[2]`
                        restart_file="restart_000000082.h5")
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_viscous_shock.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.0002576236264053728,
                            0.00014336949098706463,
                            7.189100338239794e-17,
                            0.00017369905124642074
                        ],
                        linf=[
                            0.0016731940983241156,
                            0.0010638640749656147,
                            5.59044079947959e-16,
                            0.001149532023891009
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_viscous_shock.jl (boundary_condition_do_nothing)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        boundary_conditions_parabolic=(;
                                                       x_neg = boundary_condition_parabolic,
                                                       x_pos = boundary_condition_do_nothing),
                        l2=[
                            0.0002794565402113706,
                            0.00027552504911070345,
                            3.968443509704103e-16,
                            0.0005302753211545681
                        ],
                        linf=[
                            0.001673364743408845,
                            0.0010781000885169978,
                            3.639381241716835e-15,
                            0.002801091447666215
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: P4estMesh2D: elixir_navierstokes_viscous_shock_newton_krylov.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock_newton_krylov.jl"),
                        tspan=(0.0, 0.1),
                        atol_lin_solve=1e-11,
                        rtol_lin_solve=1e-11,
                        atol_ode_solve=1e-10,
                        rtol_ode_solve=1e-10,
                        l2=[
                            3.428501006908931e-5,
                            2.5967418005884837e-5,
                            2.7084890458524478e-17,
                            2.855861765163304e-5
                        ],
                        linf=[
                            0.00018762342908784646,
                            0.0001405900207752664,
                            3.661971738081151e-16,
                            0.00014510700486747297
                        ],
                        # The GMRES solution depends on the LinearSolve.jl version and
                        # the environment at the 1e-10 level. In particular, `rho_v2`
                        # vanishes analytically for this one-dimensional setup, so its
                        # errors are pure roundoff noise that is not reproduced bitwise.
                        # Thus, relax the error tolerances.
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_SD7003airfoil.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        l2=[
                            9.292899618740586e-5,
                            0.0001350510200255721,
                            7.964907891113045e-5,
                            0.0002336568736996096
                        ],
                        linf=[
                            0.2845637352223691,
                            0.295808392241858,
                            0.19309201225626166,
                            0.7188927326929244
                        ],
                        tspan=(0.0, 5e-3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_SD7003airfoil.jl (CFL-Interval)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        l2=[
                            9.292895651912815e-5,
                            0.0001350510066877861,
                            7.964905098170568e-5,
                            0.00023365678706785303
                        ],
                        linf=[
                            0.2845614660523972,
                            0.29577255454711177,
                            0.19307666048254143,
                            0.7188872358580256
                        ],
                        tspan=(0.0, 5e-3),
                        stepsize_callback=StepsizeCallback(cfl = 2.2, interval = 5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_RAE2822airfoil_separation.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_RAE2822airfoil_separation.jl"),
                        l2=[
                            4.825503683000652e-5,
                            5.9311607940124866e-5,
                            4.275960940632988e-5,
                            0.00013394505065590255
                        ],
                        linf=[
                            1.303552961776984,
                            1.1021933490327356,
                            0.8733558923919265,
                            3.6025810936812412
                        ],
                        tspan=(0.0, 5e-5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_vortex_street.jl (Re=20)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        tspan=(0.0, 0.5),
                        Re=20, # Render flow diffusion-dominated
                        callbacks=CallbackSet(summary_callback, analysis_callback,
                                              alive_callback,
                                              StepsizeCallback(cfl = 2.3,
                                                               cfl_parabolic = 0.2)),
                        adaptive=false, # respect CFL
                        ode_alg=CKLLSRK95_4S(),
                        l2=[
                            0.011916730125703542,
                            0.02792610689451856,
                            0.019027000629638774,
                            0.11793409419172704
                        ],
                        linf=[
                            0.3546114905832565,
                            1.0152021588892746,
                            0.5811487555005529,
                            3.2073743136053725
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_vortex_street.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        l2=[
                            0.012420217727434794,
                            0.028935260981567217,
                            0.023078384429351353,
                            0.11317643179072025
                        ],
                        linf=[
                            0.4484833725983406,
                            1.268913882714608,
                            0.7071821629898418,
                            3.643975012834931
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_vortex_street.jl (GradientVariablesEntropy)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.01242797973116292,
                            0.02892502142448505,
                            0.0230829131666028,
                            0.11323126134096527
                        ],
                        linf=[
                            0.4544189333202735,
                            1.269315313304855,
                            0.7082067255956892,
                            3.6951068269010645
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_poiseuille_flow.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_poiseuille_flow.jl"),
                        l2=[
                            0.028671228188785286,
                            0.2136420195921885,
                            0.009953689550858224,
                            0.13216036594768157
                        ],
                        linf=[
                            0.30901218409540543,
                            1.3488655161645846,
                            0.1304661713119874,
                            1.2094591729756736],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_sedov_limiter_liu_zhang.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_sedov_limiter_liu_zhang.jl"),
                        l2=[
                            0.619534305,
                            0.151273076,
                            0.151273076,
                            0.287352011
                        ],
                        linf=[
                            2.60994082,
                            0.938295064,
                            0.938295064,
                            1.21442985
                        ],
                        atol=5e-2, # limiters are not smooth, so we need bigger tolerances
                        rtol=1e-2)

    # the number of iterations evaluates to 67 locally and 68 on CI
    @test length(global_limiter!.history_davis_yin_iterations) >= 67

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_kelvin_helmholtz_instability_sc_subcell.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_kelvin_helmholtz_instability_sc_subcell.jl"),
                        l2=[
                            0.1987691550257618,
                            0.1003336666735962,
                            0.1599420846677608,
                            0.07314642823482713
                        ],
                        linf=[
                            0.8901520920065688,
                            0.47421178500575756,
                            0.38859478648621326,
                            0.3247497921546598
                        ],
                        tspan=(0.0, 1.0))
    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 1.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 20_500)
end

@testitem "Parabolic2D: elixir_navierstokes_freestream_symmetry.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_symmetry.jl"),
                        l2=[
                            4.37868326434923e-15,
                            7.002449644031901e-16,
                            1.0986677074164136e-14,
                            1.213800745067394e-14
                        ],
                        linf=[
                            2.531308496145357e-14,
                            3.8367543336926215e-15,
                            4.9960036108132044e-14,
                            6.705747068735946e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_freestream_symmetry.jl (GradientVariablesEntropy)" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_symmetry.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            4.37868326434923e-15,
                            7.002449644031901e-16,
                            1.0986677074164136e-14,
                            1.213800745067394e-14
                        ],
                        linf=[
                            2.531308496145357e-14,
                            3.8367543336926215e-15,
                            4.9960036108132044e-14,
                            6.705747068735946e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_freestream_ldg.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_ldg.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            9.421780256792343e-17,
                            3.597722377997839e-15,
                            4.626519294383959e-15,
                            1.4160796757846855e-13
                        ],
                        linf=[
                            1.5543122344752192e-15,
                            1.2378986724570495e-13,
                            1.7111312367035225e-13,
                            1.5727863456049818e-11
                        ],
                        # The errors are pure round-off errors, which are sensitive
                        # to details of the compiled code (e.g., code coverage in CI)
                        atol=1.0e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_couette_flow.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_couette_flow.jl"),
                        l2=[
                            0.009585252225488753,
                            0.007939233099864973,
                            0.0007617512688442657,
                            0.027229870237669436
                        ],
                        linf=[
                            0.027230029149270862,
                            0.027230451118692933,
                            0.0038642959675975713,
                            0.04738248734987671
                        ])
    # Test `show()`
    @trixi_test_nowarn show(IOContext(stdout, :compact => false), save_solution)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: elixir_navierstokes_blast_reflective.jl" setup=[Setup, Parabolic2D] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_blast_reflective.jl"),
                        l2=[
                            0.013077652405653456,
                            0.03267271241679693,
                            0.03267271241679689,
                            0.19993587690609887
                        ],
                        linf=[
                            0.232863088636711,
                            0.5958991303183211,
                            0.5958991303183204,
                            3.0621202120365467
                        ],
                        tspan=(0.0, 0.01),
                        sol=solve(ode, ode_alg;
                                  adaptive = false, dt = 1e-4,
                                  ode_default_options()..., callback = callbacks))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic2D: TreeMesh2D: elixir_diffusion_steady_state_linear_map.jl" setup=[
    Setup,
    Parabolic2D
] tags=[:parabolic_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_steady_state_linear_map.jl"),
                        tspan=(0.0, 1.0e-4),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = 1,
                                                           extra_analysis_errors = (:l2_error_primitive,
                                                                                    :linf_error_primitive),
                                                           extra_analysis_integrals = (entropy,)),
                        l2=[2.9029827892716424e-5], linf=[0.0003022506331279151],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end
