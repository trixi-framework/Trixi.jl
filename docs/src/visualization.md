# Visualization
There are two tools provided with Trixi that allow to visualize Trixi's output
files, both of which can be found in `postprocessing/`: `trixi2vtk` and
`trixi2img`. `

## `trixi2vtk` (ParaView-based visualization)
`trixi2vtk` converts Trixi's `.h5` output files to VTK files, which can be read
by [ParaView](https://www.paraview.org) and other visualization tools. It
automatically interpolates solution data from the original quadrature node
locations to equidistant "visualization nodes" at a higher resolution, to make
up for the loss of accuracy from going from a high-order polynomial
representation to a piecewise constant representation in ParaView.

Then, to convert a file, just call `trixi2vtk` with the name of a `.h5` file as argument:
```bash
postprocessing/trixi2vtk out/solution_000000.h5
```
This allows you to generate VTK files for solution, restart and mesh files. By
default, `trixi2vtk` generates `.vtu` (unstructured VTK) files for both cell/element data (e.g.,
cell ids, element ids) and node data (e.g., solution variables). This format
visualizes each cell with the same number of nodes, independent of its size.
Alternatively, you can provide `-f vti` on the command line, which causes
`trixi2vtk` to generate `.vti` (image data VTK) files for the solution files,
while still using `.vtu` files for cell-/element-based data. In `.vti` files,
a uniform resolution is used throughout the entire domain, resulting in
different number of visualization nodes for each element.
This can be advantageous to create publication-quality images, but
increases the file size.

If you want to convert multiple solution/restart files at once, you can just supply
multiple input files on the command line. `trixi2vtk` will then also generate a
`.pvd` file, which allows ParaView to read all `.vtu`/`.vti` files at once and which
uses the `time` attribute in solution/restart files to inform ParaView about the
solution time. To list all command line options,
run `trixi2vtk --help`.

Similarly to Trixi, `trixi2vtk` supports an interactive mode that can be invoked
by running
```bash
postprocessing/trixi2vtk -i
```


## `trixi2img` (Julia-based visualization)
`trixi2img` can be used to directly convert Trixi's output files to image files,
without having to use a third-pary visualization tool such as ParaView. The
downside of this approach is that it generally takes longer to visualize the
data (especially for large files) and that it does not allow to customize the
output without having to directly edit the source code of `trixi2img`.
Currently, PNG and PDF are supported as output formats.

Then, to convert a file, just call `trixi2img` with the name of a `.h5` file as argument:
```bash
postprocessing/trixi2img out/solution_000000.h5
```
Multiple files can be converted at once by specifying more input files on the
command line.

Similarly to Trixi, `trixi2img` supports an interactive mode that can be invoked
by running
```bash
postprocessing/trixi2img -i
```

