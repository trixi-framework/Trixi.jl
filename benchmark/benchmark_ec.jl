using Printf, BenchmarkTools, Trixi

function run_benchmarks(benchmark_run; levels = 0:5, polydeg = 3)
    runtimes = zeros(length(levels))
    for (idx, initial_refinement_level) in enumerate(levels)
        result = benchmark_run(; initial_refinement_level, polydeg)
        display(result)
        runtimes[idx] = result |> median |> time # in nanoseconds
    end
    return (; levels, runtimes, polydeg)
end

function tabulate_benchmarks(args...; kwargs...)
    result = run_benchmarks(args...; kwargs...)
    println("#Elements | Runtime in seconds")
    for (level, runtime) in zip(result.levels, result.runtimes)
        @printf("%9d | %.2e\n", 4^level, 1.0e-9*runtime)
    end
    for (level, runtime) in zip(result.levels, result.runtimes)
        @printf("%.16e\n", 1.0e-9*runtime)
    end
end

function benchmark_euler(; initial_refinement_level = 1, polydeg = 3)
    γ = 1.4
    equations = CompressibleEulerEquations2D(γ)

    surface_flux = flux_ranocha
    volume_flux = flux_ranocha
    solver = DGSEM(polydeg, surface_flux, VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-2.0, -2.0)
    coordinates_max = (2.0, 2.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = initial_refinement_level,
                    n_cells_max = 100_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_weak_blast_wave,
                                        solver)

    t0 = 0.0
    u0 = compute_coefficients(t0, semi)
    du = similar(u0)

    @benchmark Trixi.rhs!($du, $u0, $semi, $t0)
end

# versioninfo(verbose=true)
@show Threads.nthreads()
tabulate_benchmarks(benchmark_euler, levels = 0:8)
