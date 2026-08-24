#!/usr/bin/env julia

const PATTERNS = ["*.jl", "*.md", "*.sh", "*.yml", "*.toml"]

function fix_whitespace(files)
    for file in files
        isfile(file) || continue
        content = read(file, String)
        isempty(content) && continue

        fixed = replace(content, "\r\n" => "\n", '\ua0' => ' ', "\t" => "    ")
        fixed = replace(fixed, r"[ \t]+$"m => "")
        fixed = rstrip(fixed) * "\n"

        content == fixed || write(file, fixed)
    end
end

function main()
    if "-h" in ARGS || "--help" in ARGS
        return nothing
    end

    if isempty(ARGS)
        files = eachline(`git ls-files -- $PATTERNS`)
    else
        files = ARGS
    end

    return fix_whitespace(files)
end

main()
