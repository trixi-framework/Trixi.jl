#!/usr/bin/env julia
#
# Script to create a sysimage with all of Trixi's dependencies and, optionally, Trixi itself.
# Note: You need to have `OrdinaryDiffEq`, `Plots`, and `Trixi2Vtk` installed as packages in the
# general environment.
#
# After the sysimage has been generated, you can use it by starting Julia with
#
#     julia --sysimage=path/to/new/sysimage
#
# Usage:
#
#     julia build_sysimage.jl
#     [TRIXI_SYSIMAGE_PATH=...] [TRIXI_SYSIMAGE_INCLUDE_TRIXI=...] julia build_sysimage.jl
#
# Optional environment variables:
#
#   TRIXI_SYSIMAGE_PATH:
#       Path where the resulting sysimage will be stored.
#       (default: `TrixiSysimage.<ext>` where `<ext>` is `.so`, `.dylib`, or `.dll`)
#
#   TRIXI_SYSIMAGE_INCLUDE_TRIXI:
#       If "no", "1", or "false" (all case-insensitive), only Trixi's direct dependencies  +
#       `OrdinaryDiffEq`, `Plots`, and `Trixi2Vtk` are stored in the sysimage. This is useful when
#       doing development with Trixi and only the startup time due to dependencies should be
#       reduced. If "yes", "1", or "true" (all case-insensitive), `Trixi` itself is included. Note
#       that in this case it is not possible to change existing functionality in Trixi anymore
#       (e.g., overwriting methods etc. will not work).
#       (default: `no`)
#
# Examples:
#
#   To include Trixi in the sysimage that should be created as `Trixi.so`, execute this script as
#   follows:
#
#       TRIXI_SYSIMAGE_PATH=Trixi.so TRIXI_SYSIMAGE_INCLUDE_TRIXI=yes julia build_sysimage.jl
#
# Special thanks to the people at the CliMA project with the ClimateMachine.jl package
# (https://github.com/CliMA/ClimateMachine.jl), from which most of this code is inspired!

using Pkg, Libdl

@info "Creating sysimage for Trixi..."
start_time = time()

# Get project directory (= repository root)
project_dir = dirname(@__DIR__)

# Add package compiler
Pkg.add("PackageCompiler")

# Activate project and install dependencies
Pkg.activate(project_dir)
Pkg.instantiate(verbose=true)
Pkg.precompile()

# Collect arguments
sysimage_path = get(ENV, "TRIXI_SYSIMAGE_PATH", joinpath(@__DIR__, "TrixiSysimage." * Libdl.dlext))
packages = Symbol[:OrdinaryDiffEq, :Plots, :Trixi2Vtk]
if lowercase(get(ENV, "TRIXI_SYSIMAGE_INCLUDE_TRIXI", "no")) in ("yes", "1", "true")
  # If Trixi is to be included, just add it to the list
  push!(packages, :Trixi)
else
  # Otherwise, figure out all direct dependencies and add them instead
  # Source: https://github.com/CliMA/ClimateMachine.jl/blob/8c57fb55acc20ee824ea37478395a7cb07c5a93c/.dev/systemimage/climate_machine_image.jl
  append!(packages, Symbol[Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep])
end
precompile_execution_file = joinpath(@__DIR__, "precompile_execution_file.jl")

# Create system image
@info "Included packages: $packages"
@info "Sysimage path: $sysimage_path"
@info "Precompile execution file: $precompile_execution_file"

using PackageCompiler
PackageCompiler.create_sysimage(
    packages,
    sysimage_path=sysimage_path,
    precompile_execution_file=precompile_execution_file,
    cpu_target=PackageCompiler.default_app_cpu_target()
)

duration = time() - start_time
@info "Done. Created sysimage in $duration seconds."
