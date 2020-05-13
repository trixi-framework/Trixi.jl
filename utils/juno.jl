# Do everything inside a let block to not introduce new variables in the REPL
let trixi_root_dir
  # Get Trixi root directory
  trixi_root_dir = dirname(@__DIR__)

  # Set project to current directory
  import Pkg
  Pkg.activate(trixi_root_dir)

  # Try to load Revise
  revise_message = "Revise initialized: changes to Trixi source code are tracked."
  try
    @eval using Revise
  catch
    # Do nothing... it probably means that Revise is not installed
    revise_message = "Revise not found (run `julia -e 'import Pkg; Pkg.add(\"Revise\")'` to install)."
  end

  # Load Trixi
  import Trixi

  # Inform user
  @info """$revise_message
           Project directory set to '$trixi_root_dir'. Adding/removing packages will only affect this project.

           Execute the following line to start a Trixi simulation:

           Trixi.run("examples/parameters.toml")
           """
end
