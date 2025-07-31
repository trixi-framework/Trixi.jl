module TestExamplesDGMulti1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_1d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "DGMulti 1D" begin
#! format: noindent

@trixi_testset "elixir_advection_gauss_sbp.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gauss_sbp.jl"),
                        cells_per_dimension=(8,),
                        l2=[2.9953644500009865e-5],
                        linf=[4.467840577382365e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_burgers_gauss_shock_capturing.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_burgers_gauss_shock_capturing.jl"),
                        cells_per_dimension=(8,), tspan=(0.0, 0.1),
                        l2=[0.445804588167854],
                        linf=[0.74780611426038])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_flux_diff.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
                        cells_per_dimension=(16,),
                        # division by sqrt(2.0) corresponds to normalization by the square root of the size of the domain
                        l2=[
                            7.853842541289665e-7,
                            9.609905503440606e-7,
                            2.832322219966481e-6
                        ] ./ sqrt(2.0),
                        linf=[
                            1.5003758788711963e-6,
                            1.802998748523521e-6,
                            4.83599270806323e-6
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shu_osher_gauss_shock_capturing.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_shu_osher_gauss_shock_capturing.jl"),
                        cells_per_dimension=(64,), tspan=(0.0, 1.0),
                        l2=[
                            1.6967151731067875,
                            6.018445633981826,
                            21.77425594743242
                        ],
                        linf=[
                            3.2229876650556477,
                            10.702690533393842,
                            38.37424900889908
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_flux_diff.jl (convergence)" begin
    using Trixi: convergence_test
    mean_convergence = convergence_test(@__MODULE__,
                                        joinpath(EXAMPLES_DIR,
                                                 "elixir_euler_flux_diff.jl"), 3)
    @test isapprox(mean_convergence[:l2],
                   [4.1558759698638434, 3.977911306037128, 4.041421206468769],
                   rtol = 0.05)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_flux_diff.jl (SBP) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
                        cells_per_dimension=(16,),
                        approximation_type=SBP(),
                        l2=[
                            6.437827414849647e-6,
                            2.1840558851820947e-6,
                            1.3245669629438228e-5
                        ],
                        linf=[
                            2.0715843751295537e-5,
                            8.519520630301258e-6,
                            4.2642194098885255e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_flux_diff.jl (FD SBP)" begin
    using Trixi: SummationByPartsOperators, derivative_operator
    global D = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                                   derivative_order = 1,
                                   accuracy_order = 4,
                                   xmin = 0.0, xmax = 1.0,
                                   N = 16)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
                        cells_per_dimension=(4,),
                        approximation_type=D,
                        l2=[
                            1.8684509287853788e-5,
                            1.0641411823379635e-5,
                            5.178010291876143e-5
                        ],
                        linf=[
                            6.933493585936645e-5,
                            3.0277366229292113e-5,
                            0.0002220020568932668
                        ])
    show(stdout, semi.solver.basis)
    show(stdout, MIME"text/plain"(), semi.solver.basis)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_modified_sod.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_1d",
                                 "elixir_euler_modified_sod.jl"),
                        cells_per_dimension=(16,),
                        l2=[0.26352391505659767, 0.4528974787813885, 0.9310255091126164],
                        linf=[
                            0.6268146194274395,
                            0.8214003799995101,
                            1.8606901431409795
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_fdsbp_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
                        l2=[
                            9.146929178341782e-7, 1.8997616876521201e-6,
                            3.991417701005622e-6
                        ],
                        linf=[
                            1.7321089882393892e-6, 3.3252888869128583e-6,
                            6.525278767988141e-6
                        ])
    show(stdout, semi.solver.basis)
    show(stdout, MIME"text/plain"(), semi.solver.basis)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti with periodic SBP unit test" begin
    using Trixi: periodic_derivative_operator, DGMulti, DGMultiMesh
    # see https://github.com/trixi-framework/Trixi.jl/pull/1013
    global D = periodic_derivative_operator(derivative_order = 1,
                                            accuracy_order = 4,
                                            xmin = -5.0,
                                            xmax = 10.0, N = 50)
    dg = DGMulti(element_type = Line(), approximation_type = D)
    mesh = DGMultiMesh(dg)
    @test mapreduce(isapprox, &, mesh.md.xyz, dg.basis.rst)
    # check to make sure nodes are rescaled to [-1, 1]
    @test minimum(dg.basis.rst[1]) ≈ -1
    @test maximum(dg.basis.rst[1])≈1 atol=0.35
end

# test non-conservative systems
@trixi_testset "elixir_euler_quasi_1d.jl (SBP) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_quasi_1d.jl"),
                        cells_per_dimension=(8,),
                        approximation_type=SBP(),
                        l2=[
                            1.633271343738687e-5,
                            9.575385661756332e-6,
                            1.2700331443128421e-5,
                            0.0
                        ],
                        linf=[
                            7.304984704381567e-5,
                            5.2365944135601694e-5,
                            6.469559594934893e-5,
                            0.0
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_quasi_1d.jl (Polynomial) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_quasi_1d.jl"),
                        cells_per_dimension=(8,),
                        approximation_type=Polynomial(),
                        l2=[
                            3.3742251708854453e-6,
                            2.9716405988822176e-6,
                            3.1641250402788772e-6,
                            1.0482169269991052e-6
                        ],
                        linf=[
                            8.056816211965412e-6,
                            6.031057946387364e-6,
                            6.90878439346676e-6,
                            1.5199471203874992e-6
                        ])
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

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
