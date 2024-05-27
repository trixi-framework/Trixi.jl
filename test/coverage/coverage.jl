# Inspired by
# - https://gitlab.com/tkpapp/GitlabJuliaDemo.jl by Tamas K. Papp
# - https://github.com/tpapp/LocalCoverage.jl by Tamas K. Papp

using Coverage

const report_dir = "coverage_report"
const lcov_info_file = "lcov.info"

# Change path to root directory
cd(joinpath(@__DIR__, "..", "..")) do
    # Process coverage files
    processed = process_folder("src")

    # Uncomment the following line once Codecov support is enabled
    # Codecov.submit_local(processed)

    # Calculate coverage
    covered_lines, total_lines = get_summary(processed)
    percentage = covered_lines / total_lines * 100

    # Print coverage in a format that can be easily parsed
    println("($(percentage)%) covered")

    # Try to generate a coverage report
    isdir(report_dir) || mkdir(report_dir)
    tracefile = joinpath(report_dir, lcov_info_file)
    Coverage.LCOV.writefile(tracefile, processed)
    branch = strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    commit = strip(read(`git rev-parse --short HEAD`, String))
    title = "commit $(commit) on branch $(branch)"
    run(`genhtml -t $(title) -o $(report_dir) $(tracefile)`)

    # Clean up .cov files
    clean_folder("src")
end
