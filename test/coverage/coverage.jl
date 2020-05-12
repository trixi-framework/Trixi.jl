# Based on https://gitlab.com/tkpapp/GitlabJuliaDemo.jl by Tamas K. Papp

using Coverage

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

  # Clean up .cov files
  clean_folder("src")
end
