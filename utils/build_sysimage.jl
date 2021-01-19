#!/usr/bin/env julia

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
packages = [:OrdinaryDiffEq, :Plots, :Trixi, :Trixi2Vtk]
sysimage_path = joinpath(@__DIR__, "TrixiSysimage." * Libdl.dlext)
precompile_execution_file = joinpath(@__DIR__, "precompile_execution_file.jl")

# Create system image
@info "Packages: $packages"
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
