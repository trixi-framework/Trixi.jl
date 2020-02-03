#!/usr/bin/env julia

# Include file with definition of main function
include("main.jl")

# Run jul1dge but handle user interrupts gracefully (Ctrl-c)
@Auxiliary.interruptable run()
