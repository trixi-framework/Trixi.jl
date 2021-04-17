
using PkgBenchmark
using Trixi

let results = judge(Trixi,
            BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=1`), # target
            BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=1`, id="main") # baseline
        )

    export_markdown(joinpath(pathof(Trixi) |> dirname |> dirname, "benchmark", "results_$(gethostname())_threads1.md"), results)
end


let results = judge(Trixi,
            BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=2`), # target
            BenchmarkConfig(juliacmd=`$(Base.julia_cmd()) --check-bounds=no --threads=2`, id="main") # baseline
        )

    export_markdown(joinpath(pathof(Trixi) |> dirname |> dirname, "benchmark", "results_$(gethostname())_threads2.md"), results)
end
