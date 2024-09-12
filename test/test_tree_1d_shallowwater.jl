module TestExamples1DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Shallow Water" begin
#! format: noindent

@trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
                        l2=[
                            0.24476140682560343,
                            0.8587309324660326,
                            0.07330427577586297
                        ],
                        linf=[
                            2.1636963952308372,
                            3.8737770522883115,
                            1.7711213427919539
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_ec.jl with initial_condition_weak_blast_wave" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
                        l2=[
                            0.39472828074570576,
                            2.0390687947320076,
                            4.1623084150546725e-10
                        ],
                        linf=[
                            0.7793741954662221,
                            3.2411927977882096,
                            7.419800190922032e-10
                        ],
                        initial_condition=initial_condition_weak_blast_wave,
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.10416666834254829,
                            1.4352935256803184e-14,
                            0.10416666834254838
                        ],
                        linf=[1.9999999999999996, 3.248036646353028e-14, 2.0],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.10416666834254835,
                            1.1891029971551825e-14,
                            0.10416666834254838
                        ],
                        linf=[2.0000000000000018, 2.4019608337954543e-14, 2.0],
                        surface_flux=(FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                                                    hydrostatic_reconstruction_audusse_etal),
                                      flux_nonconservative_audusse_etal),
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl with flux_nonconservative_wintermeyer_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.10416666834254838,
                            1.6657566141935285e-14,
                            0.10416666834254838
                        ],
                        linf=[2.0000000000000004, 3.0610625110157164e-14, 2.0],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_wintermeyer_etal),
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0022363707373868713,
                            0.01576799981934617,
                            4.436491725585346e-5
                        ],
                        linf=[
                            0.00893601803417754,
                            0.05939797350246456,
                            9.098379777405796e-5
                        ],
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.002275023323848826,
                            0.015861093821754046,
                            4.436491725585346e-5
                        ],
                        linf=[
                            0.008461451098266792,
                            0.05722331401673486,
                            9.098379777405796e-5
                        ],
                        tspan=(0.0, 0.025),
                        surface_flux=(flux_hll,
                                      flux_nonconservative_fjordholm_etal))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_nonconservative_wintermeyer_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.005774284062933275,
                            0.017408601639513584,
                            4.43649172561843e-5
                        ],
                        linf=[
                            0.01639116193303547,
                            0.05102877460799604,
                            9.098379777450205e-5
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_wintermeyer_etal),
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_source_terms_dirichlet.jl"),
                        l2=[
                            0.0022667320585353927,
                            0.01571629729279524,
                            4.4364917255842716e-5
                        ],
                        linf=[
                            0.008945234652224965,
                            0.059403165802872415,
                            9.098379777405796e-5
                        ],
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_source_terms_dirichlet.jl"),
                        l2=[
                            0.0022774071143995952,
                            0.01566214422689219,
                            4.4364917255842716e-5
                        ],
                        linf=[
                            0.008451721489057373,
                            0.05720939055279217,
                            9.098379777405796e-5
                        ],
                        surface_flux=(FluxHydrostaticReconstruction(FluxHLL(min_max_speed_naive),
                                                                    hydrostatic_reconstruction_audusse_etal),
                                      flux_nonconservative_audusse_etal),
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced_nonperiodic.jl with Dirichlet boundary" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_well_balanced_nonperiodic.jl"),
                        l2=[
                            1.725964362045055e-8,
                            5.0427180314307505e-16,
                            1.7259643530442137e-8
                        ],
                        linf=[
                            3.844551077492042e-8,
                            3.469453422316143e-15,
                            3.844551077492042e-8
                        ],
                        tspan=(0.0, 0.25),
                        surface_flux=(FluxHLL(min_max_speed_naive),
                                      flux_nonconservative_fjordholm_etal),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced_nonperiodic.jl with wall boundary" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_well_balanced_nonperiodic.jl"),
                        l2=[
                            1.7259643614361866e-8,
                            3.5519018243195145e-16,
                            1.7259643530442137e-8
                        ],
                        linf=[
                            3.844551010878661e-8,
                            9.846474508971374e-16,
                            3.844551077492042e-8
                        ],
                        tspan=(0.0, 0.25),
                        surface_flux=(FluxHLL(min_max_speed_naive),
                                      flux_nonconservative_fjordholm_etal),
                        boundary_condition=boundary_condition_slip_wall)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_shock_capturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_shock_capturing.jl"),
                        l2=[0.07424140641160326, 0.2148642632748155, 0.0372579849000542],
                        linf=[
                            1.1209754279344226,
                            1.3230788645853582,
                            0.8646939843534251
                        ],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallow_water_quasi_1d_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallow_water_quasi_1d_source_terms.jl"),
                        l2=[
                            6.37048760275098e-5,
                            0.0002745658116815704,
                            4.436491725647962e-6,
                            8.872983451152218e-6
                        ],
                        linf=[
                            0.00026747526881631956,
                            0.0012106730729152249,
                            9.098379777500165e-6,
                            1.8196759554278685e-5
                        ],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_quasi_1d_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_quasi_1d_well_balanced.jl"),
                        l2=[
                            1.4250229186905198e-14,
                            2.495109919406496e-12,
                            7.408599286788738e-17,
                            2.7205812409138776e-16
                        ],
                        linf=[
                            5.284661597215745e-14,
                            2.74056233065078e-12,
                            2.220446049250313e-16,
                            8.881784197001252e-16
                        ],
                        tspan=(0.0, 100.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_quasi_1d_discontinuous.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_quasi_1d_discontinuous.jl"),
                        l2=[
                            0.02843233740533314,
                            0.14083324483705398,
                            0.0054554472558998,
                            0.005455447255899814
                        ],
                        linf=[
                            0.26095842440037487,
                            0.45919004549253795,
                            0.09999999999999983,
                            0.10000000000000009
                        ],)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

end # module
