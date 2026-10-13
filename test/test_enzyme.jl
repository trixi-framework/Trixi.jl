# Reverse-mode automatic differentiation with Enzyme.jl.
# We check vector-Jacobian products vᵀ (∂R/∂u) w of the semidiscretization R against
# central finite differences in a random direction w. The states are perturbed randomly
# to avoid evaluating non-smooth functions such as `max(λ_ll, λ_rr)` in numerical fluxes
# or `abs(v)` exactly at their kinks, where finite differences and AD differ.

@testsnippet EnzymeSetup begin
    using Enzyme: Enzyme, Active, Const, Duplicated, Reverse, autodiff
    using LinearAlgebra: dot
    using Random: Random

    # The timers of `@trixi_timeit` are not differentiable (yet)
    Trixi.TrixiBase.disable_debug_timings()

    random_like(u) = map(x -> randn(typeof(x)), u)

    # vᵀ R(u) with R the full right-hand side (hyperbolic + parabolic)
    function vtR(u, semi, v, du, du_parabolic, t)
        Trixi.rhs_hyperbolic!(du, u, semi, t)
        if semi isa SemidiscretizationHyperbolicParabolic
            Trixi.rhs_parabolic!(du_parabolic, u, semi, t)
            du .+= du_parabolic
        end
        return dot(v, du)
    end

    # Relative error of the reverse-mode VJP compared to central finite differences in the
    # direction `w` (minimum over a few step sizes). Also returns the shadow of `semi`.
    function vjp_error(semi; t = 0.1, seed = 1)
        Random.seed!(seed)
        u = copy(Trixi.compute_coefficients(t, semi))
        u .+= 1.0e-4 .* random_like(u)
        v = random_like(u)
        w = random_like(u)
        du = similar(u)
        du_parabolic = similar(u)

        ū = zero(u)
        semī = Enzyme.make_zero(semi)
        autodiff(Reverse, vtR, Active, Duplicated(copy(u), ū), Duplicated(semi, semī),
                 Const(v), Duplicated(du, zero(du)),
                 Duplicated(du_parabolic, zero(du_parabolic)), Const(t))
        ad = dot(ū, w)

        errors = map((1.0e-5, 1.0e-6, 1.0e-7)) do ε
            u_plus = copy(u)
            u_plus .= u .+ ε .* w
            u_minus = copy(u)
            u_minus .= u .- ε .* w
            fd = (vtR(u_plus, semi, v, du, du_parabolic, t) -
                  vtR(u_minus, semi, v, du, du_parabolic, t)) / (2 * ε)
            abs(ad - fd) / abs(fd)
        end
        return minimum(errors), semī
    end
end

@testitem "Enzyme: TreeMesh 2D Euler" setup=[Setup, EnzymeSetup] tags=[:enzyme] begin
    equations = CompressibleEulerEquations2D(1.4)
    mesh = TreeMesh((0.0, 0.0), (2.0, 2.0), initial_refinement_level = 2,
                    periodicity = true)

    @timed_testset "weak form" begin
        solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
        semi = SemidiscretizationHyperbolic(mesh, equations,
                                            initial_condition_convergence_test, solver;
                                            source_terms = source_terms_convergence_test,
                                            boundary_conditions = boundary_condition_periodic)
        error, _ = vjp_error(semi)
        @test error < 1.0e-6
    end

    @timed_testset "flux differencing and derivative w.r.t. γ" begin
        solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
                       volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
        semi = SemidiscretizationHyperbolic(mesh, equations,
                                            initial_condition_convergence_test, solver;
                                            source_terms = source_terms_convergence_test,
                                            boundary_conditions = boundary_condition_periodic)
        error, semī = vjp_error(semi)
        @test error < 1.0e-6

        # The shadow of `semi` holds the derivatives w.r.t. the fields of the equations.
        # Since `inv_gamma_minus_one` is stored, d/dγ needs the chain rule.
        γ = equations.gamma
        dγ = semī.equations.gamma - semī.equations.inv_gamma_minus_one / (γ - 1)^2
        Random.seed!(1)
        u = copy(Trixi.compute_coefficients(0.1, semi))
        u .+= 1.0e-4 .* random_like(u)
        v = random_like(u)
        du = similar(u)
        function vtR_gamma(γ)
            semi_γ = SemidiscretizationHyperbolic(mesh, CompressibleEulerEquations2D(γ),
                                                  initial_condition_convergence_test,
                                                  solver;
                                                  source_terms = source_terms_convergence_test,
                                                  boundary_conditions = boundary_condition_periodic)
            return vtR(u, semi_γ, v, du, du, 0.1)
        end
        ε = 1.0e-6
        fd = (vtR_gamma(γ + ε) - vtR_gamma(γ - ε)) / (2 * ε)
        @test isapprox(dγ, fd, rtol = 1.0e-6)
    end
end

@testitem "Enzyme: P4estMesh 2D and DGMulti" setup=[Setup, EnzymeSetup] tags=[:enzyme] begin
    @timed_testset "P4estMesh 2D, curved" begin
        equations = CompressibleEulerEquations2D(1.4)
        mapping(xi, eta) = SVector(xi + 0.1 * sin(pi * eta), eta + 0.1 * sin(pi * xi))
        mesh = P4estMesh((2, 2), polydeg = 3, mapping = mapping, periodicity = true,
                         initial_refinement_level = 1)
        solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
        semi = SemidiscretizationHyperbolic(mesh, equations,
                                            initial_condition_convergence_test, solver;
                                            source_terms = source_terms_convergence_test,
                                            boundary_conditions = boundary_condition_periodic)
        error, _ = vjp_error(semi)
        @test error < 1.0e-5
    end

    @timed_testset "DGMulti 2D" begin
        equations = CompressibleEulerEquations2D(1.4)
        dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
                     surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                     volume_integral = VolumeIntegralWeakForm())
        mesh = DGMultiMesh(dg, (4, 4), periodicity = true)
        semi = SemidiscretizationHyperbolic(mesh, equations,
                                            initial_condition_convergence_test, dg;
                                            source_terms = source_terms_convergence_test,
                                            boundary_conditions = boundary_condition_periodic)
        error, _ = vjp_error(semi)
        @test error < 1.0e-5
    end
end

@testitem "Enzyme: TreeMesh 2D Navier-Stokes" setup=[Setup, EnzymeSetup] tags=[:enzyme] begin
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "tree_2d_dgsem",
                           "elixir_navierstokes_convergence.jl"),
                  initial_refinement_level = 2, tspan = (0.0, 0.0))
    error, _ = vjp_error(semi)
    @test error < 1.0e-5
end

@testitem "Enzyme: time integration with frozen CFL step size" setup=[Setup, EnzymeSetup] tags=[:enzyme] begin
    using OrdinaryDiffEqLowStorageRK: ODEProblem, CallbackSet, DiscreteCallback

    equations = CompressibleEulerEquations2D(1.4)
    mesh = TreeMesh((0.0, 0.0), (2.0, 2.0), initial_refinement_level = 2,
                    periodicity = true)
    solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
                   volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
    semi = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition_convergence_test, solver;
                                        source_terms = source_terms_convergence_test,
                                        boundary_conditions = boundary_condition_periodic)
    Random.seed!(1)
    u0 = copy(Trixi.compute_coefficients(0.0, semi))
    u0 .+= 1.0e-4 .* random_like(u0)
    c = random_like(u0)
    tspan = (0.0, 0.1)
    rhs!(du, u, semi, t) = Trixi.rhs_hyperbolic!(du, u, semi, t)

    # J(u(T)) with the step size chosen by the `StepsizeCallback`. Since `max_dt` is
    # inactive, Enzyme.jl differentiates with the step sizes of the primal run fixed.
    function J_cfl(u0, semi)
        prob = ODEProblem(rhs!, u0, tspan, semi)
        sol = Trixi.solve(prob, Trixi.CarpenterKennedy2N54(); dt = 1.0,
                          save_everystep = false,
                          callback = CallbackSet(StepsizeCallback(cfl = 0.8)))
        return 0.5 * dot(c, sol.u[end] .^ 2)
    end
    # The same with given step sizes
    function J_dts(u0, semi, dts)
        u = copy(u0)
        t = first(tspan)
        for dt in dts
            prob = ODEProblem(rhs!, u, (t, t + dt), semi)
            sol = Trixi.solve(prob, Trixi.CarpenterKennedy2N54(); dt,
                              save_everystep = false)
            u = sol.u[end]
            t += dt
        end
        return 0.5 * dot(c, u .^ 2)
    end

    # Record the step sizes of the primal run
    times = [first(tspan)]
    record = DiscreteCallback((u, t, integrator) -> true,
                              integrator -> push!(times, integrator.t);
                              save_positions = (false, false))
    Trixi.solve(ODEProblem(rhs!, u0, tspan, semi), Trixi.CarpenterKennedy2N54();
                dt = 1.0, save_everystep = false,
                callback = CallbackSet(StepsizeCallback(cfl = 0.8), record))
    dts = diff(times)
    @test J_cfl(u0, semi) ≈ J_dts(u0, semi, dts)

    ū = zero(u0)
    autodiff(Reverse, J_cfl, Active, Duplicated(copy(u0), ū),
             Duplicated(semi, Enzyme.make_zero(semi)))

    # Finite differences with the frozen step sizes
    w = random_like(u0)
    ε = 1.0e-6
    fd = (J_dts(u0 .+ ε .* w, semi, dts) - J_dts(u0 .- ε .* w, semi, dts)) / (2 * ε)
    @test isapprox(dot(ū, w), fd, rtol = 1.0e-6)
end
