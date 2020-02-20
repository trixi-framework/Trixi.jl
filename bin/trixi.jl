#!/usr/bin/env julia

# Include file with definition of main function
include("../src/main.jl")

if abspath(PROGRAM_FILE) == @__FILE__
  # Run Trixi but handle user interrupts gracefully (Ctrl-c)
  @Trixi.Auxiliary.interruptable run()
end
