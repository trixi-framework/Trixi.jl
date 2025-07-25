module TestVisualization

using Test
using Trixi
using Plots

# We use CairoMakie to avoid some CI-related issues with GLMakie. CairoMakie does not support
# interactive visualization through `iplot`, but it can be used as a testing backend for Trixi's
# Makie-based visualization.
using CairoMakie

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

# Run various visualization tests
@testset "Visualization tests" begin
#! format: noindent

# Run 2D tests with elixirs for all mesh types
test_examples_2d = Dict("TreeMesh" => ("tree_2d_dgsem",
                                       "elixir_euler_blast_wave_amr.jl"),
                        "StructuredMesh" => ("structured_2d_dgsem",
                                             "elixir_euler_source_terms_waving_flag.jl"),
                        "UnstructuredMesh" => ("unstructured_2d_dgsem",
                                               "elixir_euler_basic.jl"),
                        "P4estMesh" => ("p4est_2d_dgsem",
                                        "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                        "DGMulti" => ("dgmulti_2d", "elixir_euler_weakform.jl"))

@testset "PlotData2D, PlotDataSeries, PlotMesh with $mesh" for mesh in keys(test_examples_2d)
    # Run Trixi.jl
    directory, elixir = test_examples_2d[mesh]
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), directory, elixir),
                                   tspan = (0, 0.1))

    # Constructor tests
    if mesh == "TreeMesh"
        @test PlotData2D(sol) isa Trixi.PlotData2DCartesian
        @test PlotData2D(sol; nvisnodes = 0, grid_lines = false,
                         solution_variables = cons2cons) isa Trixi.PlotData2DCartesian
        @test Trixi.PlotData2DTriangulated(sol) isa Trixi.PlotData2DTriangulated
    else
        @test PlotData2D(sol) isa Trixi.PlotData2DTriangulated
        @test PlotData2D(sol; nvisnodes = 0, solution_variables = cons2cons) isa
              Trixi.PlotData2DTriangulated
    end
    pd = PlotData2D(sol)

    # show
    @test_nowarn_mod show(stdout, pd)
    println(stdout)

    # getindex
    @test pd["rho"] == Trixi.PlotDataSeries(pd, 1)
    @test pd["v1"] == Trixi.PlotDataSeries(pd, 2)
    @test pd["v2"] == Trixi.PlotDataSeries(pd, 3)
    @test pd["p"] == Trixi.PlotDataSeries(pd, 4)
    @test_throws KeyError pd["does not exist"]

    # convenience methods for mimicking a dictionary
    @test pd[begin] == Trixi.PlotDataSeries(pd, 1)
    @test pd[end] == Trixi.PlotDataSeries(pd, 4)
    @test length(pd) == 4
    @test size(pd) == (4,)
    @test keys(pd) == ("rho", "v1", "v2", "p")
    @test eltype(pd) <: Pair{String, <:Trixi.PlotDataSeries}
    @test [v for v in pd] == ["rho" => Trixi.PlotDataSeries(pd, 1),
        "v1" => Trixi.PlotDataSeries(pd, 2),
        "v2" => Trixi.PlotDataSeries(pd, 3),
        "p" => Trixi.PlotDataSeries(pd, 4)]

    # PlotDataSeries
    pds = pd["p"]
    @test pds.plot_data == pd
    @test pds.variable_id == 4
    @test_nowarn_mod show(stdout, pds)
    println(stdout)

    # getmesh/PlotMesh
    @test getmesh(pd) == Trixi.PlotMesh(pd)
    @test getmesh(pd).plot_data == pd
    @test_nowarn_mod show(stdout, getmesh(pd))
    println(stdout)

    @testset "2D plot recipes" begin
        pd = PlotData2D(sol)

        @test_nowarn_mod Plots.plot(sol)
        @test_nowarn_mod Plots.plot(pd)
        @test_nowarn_mod Plots.plot(pd["p"])
        @test_nowarn_mod Plots.plot(getmesh(pd))

        semi = sol.prob.p
        if mesh == "DGMulti"
            if sol.u[end] isa Trixi.VectorOfArray
                u = parent(sol.u[end])
            else
                u = sol.u[end]
            end
            scalar_data = StructArrays.component(u, 1)
            @test_nowarn_mod Plots.plot(ScalarPlotData2D(scalar_data, semi))
        else
            cache = semi.cache
            x = view(cache.elements.node_coordinates, 1, :, :, :)
            @test_nowarn_mod Plots.plot(ScalarPlotData2D(x, semi))
        end
    end

    @testset "1D plot from 2D solution" begin
        if mesh != "DGMulti"
            @testset "Create 1D plot as slice" begin
                @test_nowarn_mod PlotData1D(sol, slice = :y, point = (0.5, 0.0)) isa
                                 PlotData1D
                @test_nowarn_mod PlotData1D(sol, slice = :x, point = (0.5, 0.0)) isa
                                 PlotData1D
                pd1D = PlotData1D(sol, slice = :y, point = (0.5, 0.0))
                @test_nowarn_mod Plots.plot(pd1D)

                @testset "Create 1D plot along curve" begin
                    curve = zeros(2, 10)
                    curve[1, :] = range(-1, 1, length = 10)
                    @test_nowarn_mod PlotData1D(sol, curve = curve) isa PlotData1D
                    pd1D = PlotData1D(sol, curve = curve)
                    @test_nowarn_mod Plots.plot(pd1D)
                end
            end
        end
    end
end

@timed_testset "PlotData1D, PlotDataSeries, PlotMesh" begin
    # Run Trixi.jl
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "tree_1d_dgsem",
                                            "elixir_euler_blast_wave.jl"),
                                   tspan = (0, 0.1))

    # Constructor
    @test PlotData1D(sol) isa PlotData1D
    pd = PlotData1D(sol)

    # show
    @test_nowarn_mod show(stdout, pd)
    println(stdout)

    # getindex
    @test pd["rho"] == Trixi.PlotDataSeries(pd, 1)
    @test pd["v1"] == Trixi.PlotDataSeries(pd, 2)
    @test pd["p"] == Trixi.PlotDataSeries(pd, 3)
    @test_throws KeyError pd["does not exist"]

    # convenience methods for mimicking a dictionary
    @test pd[begin] == Trixi.PlotDataSeries(pd, 1)
    @test pd[end] == Trixi.PlotDataSeries(pd, 3)
    @test length(pd) == 3
    @test size(pd) == (3,)
    @test keys(pd) == ("rho", "v1", "p")
    @test eltype(pd) <: Pair{String, <:Trixi.PlotDataSeries}
    @test [v for v in pd] == ["rho" => Trixi.PlotDataSeries(pd, 1),
        "v1" => Trixi.PlotDataSeries(pd, 2),
        "p" => Trixi.PlotDataSeries(pd, 3)]

    # PlotDataSeries
    pds = pd["p"]
    @test pds.plot_data == pd
    @test pds.variable_id == 3
    @test_nowarn_mod show(stdout, pds)
    println(stdout)

    # getmesh/PlotMesh
    @test getmesh(pd) == Trixi.PlotMesh(pd)
    @test getmesh(pd).plot_data == pd
    @test_nowarn_mod show(stdout, getmesh(pd))
    println(stdout)

    # nvisnodes
    @test size(pd.data) == (512, 3)
    pd0 = PlotData1D(sol, nvisnodes = 0)
    @test size(pd0.data) == (256, 3)
    pd2 = PlotData1D(sol, nvisnodes = 2)
    @test size(pd2.data) == (128, 3)

    @testset "1D plot recipes" begin
        pd = PlotData1D(sol)

        @test_nowarn_mod Plots.plot(sol)
        @test_nowarn_mod Plots.plot(sol, reinterpolate = false)
        @test_nowarn_mod Plots.plot(pd)
        @test_nowarn_mod Plots.plot(pd["p"])
        @test_nowarn_mod Plots.plot(getmesh(pd))
        initial_condition_t_end(x, equations) = initial_condition(x, last(tspan),
                                                                  equations)
        @test_nowarn_mod Plots.plot(initial_condition_t_end, semi)
        @test_nowarn_mod Plots.plot((x, equations) -> x, semi)
    end

    # Fake a PlotDataXD objects to test code for plotting multiple variables on at least two rows
    # with at least one plot remaining empty
    @testset "plotting multiple variables" begin
        x = collect(0.0:0.1:1.0)
        data1d = rand(5, 11)
        variable_names = string.('a':'e')
        mesh_vertices_x1d = [x[begin], x[end]]
        fake1d = PlotData1D(x, data1d, variable_names, mesh_vertices_x1d, 0)
        @test_nowarn_mod Plots.plot(fake1d)

        y = x
        data2d = [rand(11, 11) for _ in 1:5]
        mesh_vertices_x2d = [0.0, 1.0, 1.0, 0.0]
        mesh_vertices_y2d = [0.0, 0.0, 1.0, 1.0]
        fake2d = Trixi.PlotData2DCartesian(x, y, data2d, variable_names,
                                           mesh_vertices_x2d, mesh_vertices_y2d, 0, 0)
        @test_nowarn_mod Plots.plot(fake2d)
    end
end

@timed_testset "1D plot from 2D solution" begin
    @trixi_testset "Create 1D plot along curve" begin
        using OrdinaryDiffEqSSPRK

        @testset "$MeshType" for MeshType in (P4estMesh, T8codeMesh)
            equations = CompressibleEulerEquations2D(1.4)
            solver = DGSEM(polydeg = 3,
                           surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))

            coordinates_min = (-1.0, -1.0)
            coordinates_max = (+1.0, +1.0)
            trees_per_dimension = (2, 2)
            mesh = MeshType(trees_per_dimension; polydeg = 1,
                            coordinates_min, coordinates_max,
                            initial_refinement_level = 0)

            semi = SemidiscretizationHyperbolic(mesh, equations,
                                                initial_condition_constant,
                                                solver)
            ode = semidiscretize(semi, (0.0, 0.1))
            # SSPRK43 with optimized controller of Ranocha, Dalcin, Parsani,
            # and Ketcheson (2021)
            sol = solve(ode, SSPRK43();
                        dt = 0.01, ode_default_options()...)

            x = range(0, 1, length = 100)
            curve = vcat(x', x')
            # Ensure that PlotData1D is type stable
            pd = @inferred PlotData1D(sol; curve)

            # Compare with reference data
            u_node = SVector(1.0, 0.1, -0.2, 10.0)
            val = cons2prim(u_node, equations)
            for v in axes(pd.data, 2), i in axes(pd.data, 1)
                @test isapprox(pd.data[i, v], val[v])
            end
        end
    end

    @trixi_testset "PlotData1D gives correct results" begin
        equations = CompressibleEulerEquations2D(1.4)
        solver = DGSEM(polydeg = 3,
                       surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))
        coordinates_min = (-1.0, -1.0)
        coordinates_max = (+1.0, +1.0)
        initial_refinement_level = 3

        mesh_tree = TreeMesh(coordinates_min, coordinates_max;
                             n_cells_max = 10^5,
                             initial_refinement_level)
        trees_per_dimension = (1, 1)
        mesh_p4est = P4estMesh(trees_per_dimension; polydeg = 1,
                               coordinates_min, coordinates_max,
                               initial_refinement_level)
        mesh_t8code = T8codeMesh(trees_per_dimension; polydeg = 1,
                                 coordinates_min, coordinates_max,
                                 initial_refinement_level)
        cells_per_dimension = (2, 2) .^ initial_refinement_level
        mesh_structured = StructuredMesh(cells_per_dimension,
                                         coordinates_min, coordinates_max)

        function initial_condition_taylor_green_vortex(x, t,
                                                       equations::CompressibleEulerEquations2D)
            A = 1.0 # magnitude of speed
            Ms = 0.1 # maximum Mach number

            rho = 1.0
            v1 = A * sin(x[1]) * cos(x[2])
            v2 = -A * cos(x[1]) * sin(x[2])
            p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
            p = p +
                1.0 / 16.0 * A^2 * rho *
                (cos(2 * x[1]) + 2 * cos(2 * x[2]) +
                 2 * cos(2 * x[1]) + cos(2 * x[2]))

            return prim2cons(SVector(rho, v1, v2, p), equations)
        end
        ic = initial_condition_taylor_green_vortex

        ode_tree = let mesh = mesh_tree
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        ode_p4est = let mesh = mesh_p4est
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        ode_t8code = let mesh = mesh_t8code
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        ode_structured = let mesh = mesh_structured
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        x_curve = range(0, 1, length = 1000)
        curve = vcat(x_curve', x_curve')

        @testset "TreeMesh" begin
            pd_tree = @inferred PlotData1D(ode_tree.u0, ode_tree.p; curve)
            @test pd_tree.x ≈ range(0, 1, length = length(x_curve)) * sqrt(2)

            for i in eachindex(pd_tree.x)
                x = SVector(curve[1, i],
                            curve[2, i])
                prim = SVector(pd_tree.data[i, 1],
                               pd_tree.data[i, 2],
                               pd_tree.data[i, 3],
                               pd_tree.data[i, 4])
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 1.0e-3)
            end
        end

        @testset "P4estMesh" begin
            pd_p4est = @inferred PlotData1D(ode_p4est.u0, ode_p4est.p; curve)
            @test pd_p4est.x ≈ range(0, 1, length = length(x_curve)) * sqrt(2)

            for i in eachindex(pd_p4est.x)
                x = SVector(curve[1, i],
                            curve[2, i])
                prim = SVector(pd_p4est.data[i, 1],
                               pd_p4est.data[i, 2],
                               pd_p4est.data[i, 3],
                               pd_p4est.data[i, 4])
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 1.0e-3)
            end
        end

        @testset "T8codeMesh" begin
            pd_t8code = @inferred PlotData1D(ode_t8code.u0, ode_t8code.p; curve)
            @test pd_t8code.x ≈ range(0, 1, length = length(x_curve)) * sqrt(2)

            for i in eachindex(pd_t8code.x)
                x = SVector(curve[1, i],
                            curve[2, i])
                prim = SVector(pd_t8code.data[i, 1],
                               pd_t8code.data[i, 2],
                               pd_t8code.data[i, 3],
                               pd_t8code.data[i, 4])
                # Note that the T8codeMesh uses a different algorithm to
                # compute the 1D data than the other meshes above. This is
                # less accurate so that we need to use a larger tolerance.
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 5.0e-3)
            end
        end

        @testset "StructuredMesh" begin
            pd_structured = @inferred PlotData1D(ode_structured.u0, ode_structured.p;
                                                 curve)
            @test pd_structured.x ≈ range(0, 1, length = length(x_curve)) * sqrt(2)

            for i in eachindex(pd_structured.x)
                x = SVector(curve[1, i],
                            curve[2, i])
                prim = SVector(pd_structured.data[i, 1],
                               pd_structured.data[i, 2],
                               pd_structured.data[i, 3],
                               pd_structured.data[i, 4])
                # Note that the StructuredMesh uses a different algorithm to
                # compute the 1D data than the other meshes above. This is
                # less accurate so that we need to use a larger tolerance.
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 5.0e-3)
            end
        end
    end
end

@timed_testset "PlotData1D (DGMulti)" begin
    # Test two different approximation types since these use different memory layouts:
    # - structure of arrays for `Polynomial()`
    # - array of structures for `SBP()`
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "dgmulti_1d",
                                            "elixir_euler_flux_diff.jl"),
                                   tspan = (0.0, 0.0),
                                   approximation_type = Polynomial())
    @test PlotData1D(sol) isa PlotData1D
    initial_condition_t_end(x, equations) = initial_condition(x, last(tspan), equations)
    @test_nowarn_mod Plots.plot(initial_condition_t_end, semi)
    @test_nowarn_mod Plots.plot((x, equations) -> x, semi)

    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "dgmulti_1d",
                                            "elixir_euler_flux_diff.jl"),
                                   tspan = (0.0, 0.0),
                                   approximation_type = SBP())
    @test PlotData1D(sol) isa PlotData1D
    @test_nowarn_mod Plots.plot(initial_condition_t_end, semi)
    @test_nowarn_mod Plots.plot((x, equations) -> x, semi)
end

@timed_testset "1D plot recipes (StructuredMesh)" begin
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "structured_1d_dgsem",
                                            "elixir_euler_source_terms.jl"),
                                   tspan = (0.0, 0.0))

    pd = PlotData1D(sol)
    initial_condition_t_end(x, equations) = initial_condition(x, last(tspan), equations)
    @test_nowarn_mod Plots.plot(sol)
    @test_nowarn_mod Plots.plot(pd)
    @test_nowarn_mod Plots.plot(pd["p"])
    @test_nowarn_mod Plots.plot(sol.u[end], semi)
    @test_nowarn_mod Plots.plot(initial_condition_t_end, semi)
    @test_nowarn_mod Plots.plot((x, equations) -> x, semi)
end

@timed_testset "plot time series" begin
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "tree_2d_dgsem",
                                            "elixir_acoustics_gaussian_source.jl"),
                                   tspan = (0, 0.05))

    @test_nowarn_mod Plots.plot(time_series, 1)
    @test PlotData1D(time_series, 1) isa PlotData1D
end

@timed_testset "adapt_to_mesh_level" begin
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "tree_2d_dgsem",
                                            "elixir_advection_basic.jl"),
                                   analysis_callback = Trixi.TrivialCallback())
    @test adapt_to_mesh_level(sol, 5) isa Tuple

    u_ode_level5, semi_level5 = adapt_to_mesh_level(sol, 5)
    u_ode_level4, semi_level4 = adapt_to_mesh_level(u_ode_level5, semi_level5, 4)
    @test isapprox(sol.u[end], u_ode_level4, atol = 1e-13)

    @test adapt_to_mesh_level!(sol, 5) isa Tuple
    @test isapprox(sol.u[end], u_ode_level5, atol = 1e-13)
end

@timed_testset "plot 3D" begin
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "tree_3d_dgsem",
                                            "elixir_advection_basic.jl"),
                                   analysis_callback = Trixi.TrivialCallback(),
                                   initial_refinement_level = 1)
    @test PlotData2D(sol) isa Trixi.PlotData2DCartesian
    @test PlotData2D(sol, slice = :yz) isa Trixi.PlotData2DCartesian
    @test PlotData2D(sol, slice = :xz) isa Trixi.PlotData2DCartesian

    @testset "1D plot from 3D solution and Tree-mesh" begin
        @testset "Create 1D plot as slice" begin
            @test_nowarn_mod PlotData1D(sol) isa PlotData1D
            pd1D = PlotData1D(sol)
            @test_nowarn_mod Plots.plot(pd1D)
            @test_nowarn_mod PlotData1D(sol, slice = :y, point = (0.5, 0.3, 0.1)) isa
                             PlotData1D
            @test_nowarn_mod PlotData1D(sol, slice = :z, point = (0.1, 0.3, 0.3)) isa
                             PlotData1D
        end

        @testset "Create 1D plot along curve" begin
            curve = zeros(3, 10)
            curve[1, :] = range(-1.0, -0.5, length = 10)
            @test_nowarn_mod PlotData1D(sol, curve = curve) isa PlotData1D
            pd1D = PlotData1D(sol, curve = curve)
            @test_nowarn_mod Plots.plot(pd1D)
        end
    end

    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "structured_3d_dgsem",
                                            "elixir_advection_basic.jl"))

    @testset "1D plot from 3D solution and general mesh" begin
        @testset "Create 1D plot as slice" begin
            @test_nowarn_mod PlotData1D(sol) isa PlotData1D
            pd1D = PlotData1D(sol)
            @test_nowarn_mod Plots.plot(pd1D)
            @test_nowarn_mod PlotData1D(sol, slice = :y, point = (0.5, 0.3, 0.1)) isa
                             PlotData1D
            @test_nowarn_mod PlotData1D(sol, slice = :z, point = (0.1, 0.3, 0.3)) isa
                             PlotData1D
        end

        @testset "Create 1D plot along curve" begin
            curve = zeros(3, 10)
            curve[1, :] = range(-1.0, 1.0, length = 10)
            @test_nowarn_mod PlotData1D(sol, curve = curve) isa PlotData1D
            pd1D = PlotData1D(sol, curve = curve)
            @test_nowarn_mod Plots.plot(pd1D)
        end
    end

    @timed_testset "1D plot from 3D solution on P4estMesh and T8codeMesh" begin
        @trixi_testset "Create 1D plot along curve" begin
            using OrdinaryDiffEqSSPRK

            @testset "$MeshType" for MeshType in (P4estMesh, T8codeMesh)
                equations = CompressibleEulerEquations3D(1.4)
                solver = DGSEM(polydeg = 3,
                               surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))

                coordinates_min = (-1.0, -1.0, -1.0)
                coordinates_max = (+1.0, +1.0, +1.0)
                trees_per_dimension = (2, 2, 2)
                mesh = MeshType(trees_per_dimension; polydeg = 1,
                                coordinates_min, coordinates_max,
                                initial_refinement_level = 0)

                semi = SemidiscretizationHyperbolic(mesh, equations,
                                                    initial_condition_constant,
                                                    solver)
                ode = semidiscretize(semi, (0.0, 0.1))
                # SSPRK43 with optimized controller of Ranocha, Dalcin, Parsani,
                # and Ketcheson (2021)
                sol = solve(ode, SSPRK43();
                            dt = 0.01, ode_default_options()...)

                x = range(0, 1, length = 100)
                curve = vcat(x', x', x')
                # Ensure that PlotData1D is type stable
                pd = @inferred PlotData1D(sol; curve)

                # Compare with reference data
                u_node = SVector(1.0, 0.1, -0.2, 0.7, 10.0)
                val = cons2prim(u_node, equations)
                for v in axes(pd.data, 2), i in axes(pd.data, 1)
                    @test isapprox(pd.data[i, v], val[v])
                end
            end
        end
    end

    @trixi_testset "PlotData1D gives correct results" begin
        equations = CompressibleEulerEquations3D(1.4)
        solver = DGSEM(polydeg = 3,
                       surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))
        coordinates_min = (-1.0, -1.0, -1.0)
        coordinates_max = (+1.0, +1.0, +1.0)
        initial_refinement_level = 3

        mesh_tree = TreeMesh(coordinates_min, coordinates_max;
                             n_cells_max = 10^6,
                             initial_refinement_level)
        trees_per_dimension = (1, 1, 1)
        mesh_p4est = P4estMesh(trees_per_dimension; polydeg = 1,
                               coordinates_min, coordinates_max,
                               initial_refinement_level)
        mesh_t8code = T8codeMesh(trees_per_dimension; polydeg = 1,
                                 coordinates_min, coordinates_max,
                                 initial_refinement_level)
        cells_per_dimension = (2, 2, 2) .^ initial_refinement_level
        mesh_structured = StructuredMesh(cells_per_dimension,
                                         coordinates_min, coordinates_max)

        function initial_condition_taylor_green_vortex(x, t,
                                                       equations::CompressibleEulerEquations3D)
            A = 1.0 # magnitude of speed
            Ms = 0.1 # maximum Mach number

            rho = 1.0
            v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
            v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
            v3 = 0.0
            p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
            p = p +
                1.0 / 16.0 * A^2 * rho *
                (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) +
                 2 * cos(2 * x[1]) +
                 cos(2 * x[2]) * cos(2 * x[3]))

            return prim2cons(SVector(rho, v1, v2, v3, p), equations)
        end
        ic = initial_condition_taylor_green_vortex

        ode_tree = let mesh = mesh_tree
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        ode_p4est = let mesh = mesh_p4est
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        ode_t8code = let mesh = mesh_t8code
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        ode_structured = let mesh = mesh_structured
            semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)
            ode = semidiscretize(semi, (0.0, 0.1))
        end

        x_curve = range(0, 1, length = 1000)
        curve = vcat(x_curve', x_curve', x_curve')

        @testset "TreeMesh" begin
            pd_tree = @inferred PlotData1D(ode_tree.u0, ode_tree.p; curve)
            @test pd_tree.x ≈ range(0, 1, length = length(x_curve)) * sqrt(3)

            for i in eachindex(pd_tree.x)
                x = SVector(curve[1, i],
                            curve[2, i],
                            curve[3, i])
                prim = SVector(pd_tree.data[i, 1],
                               pd_tree.data[i, 2],
                               pd_tree.data[i, 3],
                               pd_tree.data[i, 4],
                               pd_tree.data[i, 5])
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 1.0e-3)
            end
        end

        @testset "P4estMesh" begin
            pd_p4est = @inferred PlotData1D(ode_p4est.u0, ode_p4est.p; curve)
            @test pd_p4est.x ≈ range(0, 1, length = length(x_curve)) * sqrt(3)

            for i in eachindex(pd_p4est.x)
                x = SVector(curve[1, i],
                            curve[2, i],
                            curve[3, i])
                prim = SVector(pd_p4est.data[i, 1],
                               pd_p4est.data[i, 2],
                               pd_p4est.data[i, 3],
                               pd_p4est.data[i, 4],
                               pd_p4est.data[i, 5])
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 1.0e-3)
            end
        end

        @testset "T8codeMesh" begin
            pd_t8code = @inferred PlotData1D(ode_t8code.u0, ode_t8code.p; curve)
            @test pd_t8code.x ≈ range(0, 1, length = length(x_curve)) * sqrt(3)

            for i in eachindex(pd_t8code.x)
                x = SVector(curve[1, i],
                            curve[2, i],
                            curve[3, i])
                prim = SVector(pd_t8code.data[i, 1],
                               pd_t8code.data[i, 2],
                               pd_t8code.data[i, 3],
                               pd_t8code.data[i, 4],
                               pd_t8code.data[i, 5])
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                @test isapprox(prim, cons2prim(u, equations), atol = 1.0e-3)
            end
        end

        @testset "StructuredMesh" begin
            pd_structured = @inferred PlotData1D(ode_structured.u0, ode_structured.p;
                                                 curve)
            @test pd_structured.x ≈ range(0, 1, length = length(x_curve)) * sqrt(3)

            for i in eachindex(pd_structured.x)
                x = SVector(curve[1, i],
                            curve[2, i],
                            curve[3, i])
                prim = SVector(pd_structured.data[i, 1],
                               pd_structured.data[i, 2],
                               pd_structured.data[i, 3],
                               pd_structured.data[i, 4],
                               pd_structured.data[i, 5])
                u = initial_condition_taylor_green_vortex(x, 0.0, equations)
                # Note that the StructuredMesh uses a different algorithm to
                # compute the 1D data than the other meshes. This is less accurate
                # so that we need to use a larger tolerance.
                @test isapprox(prim, cons2prim(u, equations), atol = 5.0e-3)
            end
        end
    end
end

@timed_testset "plotting TimeIntegratorSolution" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_hypdiff_lax_friedrichs.jl"),
                        maxiters=1, analysis_callback=Trixi.TrivialCallback(),
                        initial_refinement_level=1)
    @test_nowarn_mod Plots.plot(sol)
end

@timed_testset "VisualizationCallback" begin
    # To make CI tests work, disable showing a plot window with the GR backend of the Plots package
    # Xref: https://github.com/jheinen/GR.jl/issues/278
    # Xref: https://github.com/JuliaPlots/Plots.jl/blob/8cc6d9d48755ba452a2835f9b89d3880e9945377/test/runtests.jl#L103
    if !isinteractive()
        restore = get(ENV, "GKSwstype", nothing)
        ENV["GKSwstype"] = "100"
    end

    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "tree_2d_dgsem",
                                            "elixir_advection_amr_visualization.jl"),
                                   visualization = VisualizationCallback(semi;
                                                                         interval = 20,
                                                                         clims = (0, 1),
                                                                         plot_creator = Trixi.save_plot),
                                   tspan = (0.0, 3.0))

    @testset "elixir_advection_amr_visualization.jl with save_plot" begin
        @test isfile(joinpath(outdir, "solution_000000000.png"))
        @test isfile(joinpath(outdir, "solution_000000020.png"))
        @test isfile(joinpath(outdir, "solution_000000022.png"))
    end

    @testset "show" begin
        @test_nowarn_mod show(stdout, visualization)
        println(stdout)

        @test_nowarn_mod show(stdout, "text/plain", visualization)
        println(stdout)
    end

    # Restore GKSwstype to previous value (if it was set)
    if !isinteractive()
        if isnothing(restore)
            delete!(ENV, "GKSwstype")
        else
            ENV["GKSwstype"] = restore
        end
    end
end

@timed_testset "Makie visualization tests for UnstructuredMesh2D" begin
    @test_nowarn_mod trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "unstructured_2d_dgsem",
                                            "elixir_euler_wall_bc.jl"))

    # test interactive surface plot
    @test_nowarn_mod Trixi.iplot(sol)

    # also test when using PlotData2D object
    @test PlotData2D(sol) isa Trixi.PlotData2DTriangulated
    @test_nowarn_mod Makie.plot(PlotData2D(sol))

    # test interactive ScalarPlotData2D plotting
    semi = sol.prob.p
    x = view(semi.cache.elements.node_coordinates, 1, :, :, :) # extracts the node x coordinates
    y = view(semi.cache.elements.node_coordinates, 2, :, :, :) # extracts the node x coordinates
    @test_nowarn_mod iplot(ScalarPlotData2D(x .+ y, semi), plot_mesh = true)

    # test heatmap plot
    @test_nowarn_mod Makie.plot(sol, plot_mesh = true)

    # test unpacking/iteration for FigureAndAxes
    fa = Makie.plot(sol)
    fig, axes = fa
    @test_nowarn_mod Base.show(fa) === nothing
    @test_nowarn_mod typeof(fig) <: Makie.Figure
    @test_nowarn_mod typeof(axes) <: AbstractArray{<:Makie.Axis}

    # test plotting of constant solutions with Makie
    # related issue: https://github.com/MakieOrg/Makie.jl/issues/931
    for i in eachindex(sol.u)
        fill!(sol.u[i], one(eltype(sol.u[i])))
    end
    @test_nowarn_mod Trixi.iplot(sol)
end
end

end #module
