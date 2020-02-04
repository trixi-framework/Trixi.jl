#!/usr/bin/env julia

using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using JuliaFormatter: format


function main()
  # Parse command line arguments
  args = parse_commandline_arguments()

  # Call formatter with our default options
  format(args["path"],
         overwrite = true,
         verbose = true,
         indent = 2,
         margin = 100,
         always_for_in = true)
end


function parse_commandline_arguments()
  s = ArgParseSettings()
  @add_arg_table s begin
    "path"
      help = ("Name of file or folder to format. If PATH is a folder, "
              * "its contents are examined recursively and all `.jl` files are formatted.")
      arg_type = String
      required = true
      nargs = '+'
  end

  return parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
