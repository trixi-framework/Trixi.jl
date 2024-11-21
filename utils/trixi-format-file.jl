#!/usr/bin/env julia

using Pkg
Pkg.activate(; temp = true, io = devnull)
Pkg.add(PackageSpec(name = "JuliaFormatter", version = "1.0.60"); preserve = PRESERVE_ALL,
        io = devnull)

using JuliaFormatter: format_file

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

    file_list = ARGS
    if isempty(ARGS)
        exit(0)
    end
    non_formatted_files = Vector{String}()
    for file in file_list
        println("Checking file " * file)
        if !format_file(file)
            push!(non_formatted_files, file)
        end
    end
    if isempty(non_formatted_files)
        exit(0)
    else
        @error "Some files have not been formatted! Formatting has been applied, run 'git add -p' to update changes."
        for file in non_formatted_files
            println(file)
        end
        exit(1)
    end
end

main()
