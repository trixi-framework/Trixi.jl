#src # Unstructured meshes with HOHQMesh.jl

# Trixi.jl supports numerical approximations on unstructured quadrilateral meshes
# with the [`UnstructuredMesh2D`](@ref) mesh type.

# The purpose of this tutorial is to demonstrate how to use the `UnstructuredMesh2D`
# functionality of Trixi.jl. This begins by running and visualizing an available unstructured
# quadrilateral mesh example. Then, the tutorial will demonstrate how to
# conceptualize a problem with curved boundaries, generate
# a curvilinear mesh using the available software in the Trixi.jl ecosystem,
# and then run a simulation using Trixi.jl on said mesh.

# Unstructured quadrilateral meshes can be made
# with the [High-Order Hex-Quad Mesh (HOHQMesh) generator](https://github.com/trixi-framework/HOHQMesh)
# created and developed by David Kopriva.
# HOHQMesh is a mesh generator specifically designed for spectral element methods.
# It provides high-order boundary curve information (needed to accurately set boundary conditions)
# and elements can be larger (due to the high accuracy of the spatial approximation)
# compared to traditional finite element mesh generators.
# For more information about the design and features of HOHQMesh one can refer to its
# [official documentation](https://trixi-framework.github.io/HOHQMesh/).

# HOHQMesh is incorporated into the Trixi.jl framework via the registered Julia package
# [HOHQMesh.jl](https://github.com/trixi-framework/HOHQMesh.jl).
# This package provides a Julia wrapper for the HOHQMesh generator that allows users to easily create mesh
# files without the need to build HOHQMesh from source. To install the HOHQMesh package execute
# ```julia
# import Pkg; Pkg.add("HOHQMesh")
# ```
# Now we are ready to generate an unstructured quadrilateral mesh that can be used by Trixi.jl.

# ## Running and visualizing an unstructured simulation

# Trixi.jl supports solving hyperbolic problems on several mesh types.
# There is a default example for this mesh type that can be executed by

using Trixi
rm("out", force = true, recursive = true) #hide #md
# Test 1: current state

redirect_stdio(stdout = devnull, stderr = devnull) do # code that prints annoying stuff we don't want to see here #hide #md
    trixi_include(default_example_unstructured())
end #hide #md

# Test 1: redirect_stdio and ; (Formatter hates it)
redirect_stdio(stdout = devnull, stderr = devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi_include(default_example_unstructured());
end #hide #md

# Test 2: redirect_stdio and no ; (Formatter hates it)
redirect_stdio(stdout = devnull, stderr = devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi_include(default_example_unstructured())
end #hide #md

# Test 3: no ;
trixi_include(default_example_unstructured())

# Test 4: ;
trixi_include(default_example_unstructured());

# Test 5: ; and nothing
trixi_include(default_example_unstructured());
nothing; #hide

# Test 6: no ; and nothing;
trixi_include(default_example_unstructured())
nothing; #hide

# Test 7: ; and nothing
trixi_include(default_example_unstructured());
nothing #hide

# Test 8: no ; and nothing
trixi_include(default_example_unstructured())
nothing #hide

# This will compute a smooth, manufactured solution test case for the 2D compressible Euler equations
# on the curved quadrilateral mesh described in the
# [Trixi.jl documentation](https://trixi-framework.github.io/TrixiDocumentation/stable/meshes/unstructured_quad_mesh/).

# Apart from the usual error and timing output provided by the Trixi.jl run, it is useful to visualize and inspect
# the solution. One option available in the Trixi.jl framework to visualize the solution on
# unstructured quadrilateral meshes is post-processing the
# Trixi.jl output file(s) with the [`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl) tool
# and plotting them with [ParaView](https://www.paraview.org/download/).

# To convert the HDF5-formatted `.h5` output file(s) from Trixi.jl into VTK format execute the following

using Trixi2Vtk
# 1
trixi2vtk("out/solution_000000180.h5", output_directory = "out")
# 2
trixi2vtk("out/solution_000000180.h5", output_directory = "out");
# 3
trixi2vtk("out/solution_000000180.h5", output_directory = "out")
nothing; #hide
# 4
trixi2vtk("out/solution_000000180.h5", output_directory = "out");
nothing; #hide
# 5
trixi2vtk("out/solution_000000180.h5", output_directory = "out")
nothing #hide
# 6
trixi2vtk("out/solution_000000180.h5", output_directory = "out");
nothing #hide

# Note this step takes about 15-30 seconds as the package `Trixi2Vtk` must be precompiled and executed for the first time
# in your REPL session. The `trixi2vtk` command above will convert the solution file at the final time into a `.vtu` file
# which can be read in and visualized with ParaView. Optional arguments for `trixi2vtk` are: (1) Pointing to the `output_directory`
# where the new files will be saved; it defaults to the current directory. (2) Specifying a higher number of
# visualization nodes. For instance, if we want to use 12 uniformly spaced nodes for visualization we can execute

trixi2vtk("out/solution_000000180.h5", output_directory = "out", nvisnodes = 12);

# By default `trixi2vtk` sets `nvisnodes` to be the same as the number of nodes specified in
# the `elixir` file used to run the simulation.

# Finally, if you want to convert all the solution files to VTK execute

redirect_stdio(stdout = devnull, stderr = devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi2vtk("out/solution_000*.h5", output_directory = "out", nvisnodes = 12)
end #hide #md

# then it is possible to open the `.pvd` file with ParaView and create a video of the simulation.
