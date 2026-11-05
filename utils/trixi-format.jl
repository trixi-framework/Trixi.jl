#!/usr/bin/env julia

using Pkg
Pkg.activate(; temp = true, io = devnull)
Pkg.add(PackageSpec(name = "JuliaFormatter", version = "1.0.60"); preserve = PRESERVE_ALL,
        io = devnull)

using JuliaFormatter: format

function main()
    # Show help
    if "-h" in ARGS || "--help" in ARGS
        println("usage: trixi-format.jl PATH [PATH...]")
        println()
        println("positional arguments:")
        println()
        println("    PATH        One or more paths (directories or files) to format. Default: '.'")
        return nothing
    end

    # Set default path if none is given on command line
    if isempty(ARGS)
        paths = String["."]
    else
        paths = ARGS
    end

    return format(paths)
end

main()
