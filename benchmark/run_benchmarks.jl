using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
Pkg.instantiate()

using PkgBenchmark
using Trixi

let results = judge(Trixi,
                    BenchmarkConfig(juliacmd = `$(Base.julia_cmd()) --check-bounds=no --threads=1`), # target
                    BenchmarkConfig(juliacmd = `$(Base.julia_cmd()) --check-bounds=no --threads=1`,
                                    id = "main"))
    export_markdown(pkgdir(Trixi, "benchmark", "results_$(gethostname())_threads1.md"),
                    results)
end

let results = judge(Trixi,
                    BenchmarkConfig(juliacmd = `$(Base.julia_cmd()) --check-bounds=no --threads=2`), # target
                    BenchmarkConfig(juliacmd = `$(Base.julia_cmd()) --check-bounds=no --threads=2`,
                                    id = "main"))
    export_markdown(pkgdir(Trixi, "benchmark", "results_$(gethostname())_threads2.md"),
                    results)
end
