#!/usr/bin/env julia

# Include file with definition of main function
include("../src/main.jl")

# Run jul1dge but handle user interrupts gracefully (Ctrl-c)
@Jul1dge.Auxiliary.interruptable run()
