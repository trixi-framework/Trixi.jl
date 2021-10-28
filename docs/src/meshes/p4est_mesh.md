# P4est-based mesh

The [`P4estMesh`](@ref) is an unstructured, curvilinear, nonconforming
mesh type for quadrilateral (2D) and hexahedral (3D) cells.
It supports quadtree/octree-based adaptive mesh refinement (AMR) via
the C library [p4est](https://github.com/cburstedde/p4est). See
[`AMRCallback`](@ref) for further information.

Due to its curvilinear nature, (numerical) fluxes need to implement methods
dispatching on the `normal::AbstractVector`. Rotationally invariant equations
such as the compressible Euler equations can use [`FluxRotated`](@ref) to
wrap numerical fluxes implemented only for Cartesian meshes. This simplifies
the re-use of existing functionality for the [`TreeMesh`](@ref) but is usually
less efficient, cf. [PR #550](https://github.com/trixi-framework/Trixi.jl/pull/550).

## Construction of P4estMesh from an Abaqus file

One available option to construct a [`P4estMesh`](@ref) is to read in an Abaqus (`.inp`) mesh file.
We briefly describe the structure of this file, the conventions it uses, and how the mesh file
is parsed to create an initial unstructured, curvilinear, and conforming mesh.

For this discussion we use the following two-dimensional unstructured curved mesh with three elements:

![abaqus-mesh-docs](https://user-images.githubusercontent.com/25242486/139241997-88e70a01-286f-4cee-80b1-2fd83c60bcca.png)

We note that the node and element connectivity information parsed from the Abaqus file creates
a straight sided (linear) mesh.
From this linear mesh there are two strategies available to make the mesh curvilinear:

1. Apply a `mapping` function to describe a transformation of the linear mesh to another
   physical domain. The mapping is approximated using interpolation polynomial of a user
   specified polynomial degree. The default value of this polynomial degree is `1` that
   corresponds to an uncurved geometry.
2. High-order boundary information is available in the `.inp` mesh file because it was created
   with the [`HOHQMesh`](https://github.com/trixi-framework/HOHQMesh.jl) generator.
   This information is used to create appropriate transfinite mappings during the mesh construction.

We divide our discussion into two parts. The first part discusses the standard node and element information
contained in the `.inp` mesh file. The second part specifically deals with the mesh file parsing of an Abaqus
file created by HOHQMesh.

### Mesh file header

A `.inp` mesh file typically begins with a `*Heading`.
Though *optional*, the `*Heading` is helpful to give users some information about the mesh described by the mesh file.
In particular, a `.inp` mesh file created with HOHQMesh will contain the header
```
*Heading
 File created by HOHQMesh
```
This heading is used to indicate to the mesh constructor which of the above mapping strategies to apply in order to
create a curvilinear mesh.
If the Abaqus file header is **not** present then the `P4estMesh` is created with the first strategy above.

### List of corner nodes

Next, prefaced with `*NODE`, comes a list of the physical `(x,y,z)` coordinates of all the corners.
The first integer in the list of the corner nodes provides the node id number.
Thus, for the two-dimensional example mesh this block of corner node information is
```
*NODE
1, 1.0, -1.0, 0.0
2, 3.0,  0.0, 0.0
3, 1.0,  1.0, 0.0
4, 2.0,  0.0, 0.0
5, 0.0,  0.0, 0.0
6, 3.0,  1.0, 0.0
7, 3.0, -1.0, 0.0
```

### List of elements

The element connectivity is given after the list of corner nodes. The header for this information block is
```
*ELEMENT, type=CPS4, ELSET=Surface1
```
The Abaqus element type `CPS4` corresponds to a quadrilateral element.
Each quadrilateral element in the unstructured mesh is dictated by four corner points with indexing
taken from the numbering given by the corner list above.
The elements connect a set of four corner points (starting from the bottom left) in an anti-clockwise fashion;
making the element *right-handed*.
This is element handedness is indicated using the circular arrow in the figure above.
Just as with the corner list, the first integer in the element connectivity list indicates the element id number.
Thus, the element connectivity list for the three element example mesh is
```
*ELEMENT, type=CPS4, ELSET=Surface1
1, 5, 1, 4, 3
2, 4, 2, 6, 3
3, 7, 2, 4, 1
```

*As a short note, in a three-dimensional Abaqus file the element connectivity would be given in terms of the
eight corner nodes that define a hexahedron. Also, the header of the element section changes to be*
```
*ELEMENT, type=C3D8, ELSET=Volume1
```
*The Abaqus element type `C3D8` corresponds to a hexahedral element.*

### Element neighbor connectivity

The construction of the element nieghbor ids and identifying physical boundary surfaces is done using functionality
directly from the [p4est](https://github.com/cburstedde/p4est) library.
For example, the neighbor connectivity is created in the mesh constructor using the wrapper `read_inp_p4est` function.

### HOHQMesh boundary information

If present, any additional information in the mesh file that was created by `HOHQMesh` is prefaced with
"`** `" to make it an Abaqus comment.
This ensures that the read in of the file by a standard Abaqus file parser,
as done in the `read_inp_p4est` function, is done correctly.

The high-order, curved boundary information and labels of the physical boundary created by `HOHQMesh`
is found below the comment line
```
** ***** HOHQMesh boundary information ***** **
```
Next comes the *polynomial degree* that the mesh will use to represent any curved sides
```
** mesh polynomial degree = 8
```

The mesh file then, again, provides the element connectivity as well as information for
curved surfaces either interior to the domain or along the physical boundaries.
A set of check digits are included directly below the four corner indexes to indicate whether
the local surface index (`-y`, `+x`, `+y`, or `-x`) within the element is
straight sided, `0`, or is curved, `1`.
If the local surface is straight sided no additional information is necessary during the mesh file read in.
But for any curved surfaces the mesh file provides `(x,y,z)` coordinate values in order to construct an
interpolant of this surface with the mesh polynomial order at the Chebyshev-Gauss-Lobatto
nodes. This list of `(x,y,z)` data will be given in the direction of the local coordinate system.
Given below is the element curvature information for the example mesh:
```
**  5 1 4 3
**  0 0 1 1
**   1.000000000000000   1.000000000000000   0.0
**   1.024948365654583   0.934461926834452   0.0
**   1.116583018200151   0.777350964621867   0.0
**   1.295753434047077   0.606254343587194   0.0
**   1.537500000000000   0.462500000000000   0.0
**   1.768263070247418   0.329729152118310   0.0
**   1.920916981799849   0.185149035378133   0.0
**   1.986035130050921   0.054554577460044   0.0
**   2.000000000000000                 0.0   0.0
**                 0.0                 0.0   0.0
**   0.035513826946206   0.105291711848750   0.0
**   0.148591270347399   0.317731556850611   0.0
**   0.340010713990041   0.452219430075470   0.0
**   0.575000000000000   0.462500000000000   0.0
**   0.788022294598950   0.483764065630034   0.0
**   0.926408729652601   0.644768443149389   0.0
**   0.986453164464803   0.883724792445746   0.0
**   1.000000000000000   1.000000000000000   0.0
**  4 2 6 3
**  0 0 0 1
**   2.000000000000000                 0.0   0.0
**   1.986035130050921   0.054554577460044   0.0
**   1.920916981799849   0.185149035378133   0.0
**   1.768263070247418   0.329729152118310   0.0
**   1.537500000000000   0.462500000000000   0.0
**   1.295753434047077   0.606254343587194   0.0
**   1.116583018200151   0.777350964621867   0.0
**   1.024948365654583   0.934461926834452   0.0
**   1.000000000000000   1.000000000000000   0.0
**  7 2 4 1
**  0 0 0 0
```

The last piece of information provided by `HOHQMesh` are labels for the different surfaces of an element.
These labels are useful to set boundary conditions along physical surfaces.
The labels can be short descriptive words.
The label `---` indicates an internal surface where no boundary condition is required.

It is important to note that these labels are given in the following order according to the
local surface index `-x` `+x` `-y` `+y` as required by the [p4est](https://github.com/cburstedde/p4est) library.
```
**  Bezier --- Slant ---
**  --- Right --- Top
**  Bottom --- Right ---
```

For completeness, we provide the entire Abaqus mesh file for the example mesh in the figure above:
```
*Heading
 File created by HOHQMesh
*NODE
1, 1.0, -1.0, 0.0
2, 3.0,  0.0, 0.0
3, 1.0,  1.0, 0.0
4, 2.0,  0.0, 0.0
5, 0.0,  0.0, 0.0
6, 3.0,  1.0, 0.0
7, 3.0, -1.0, 0.0
*ELEMENT, type=CPS4, ELSET=Surface1
1, 5, 1, 4, 3
2, 4, 2, 6, 3
3, 7, 2, 4, 1
** ***** HOHQMesh boundary information ***** **
** mesh polynomial degree = 8
**  5 1 4 3
**  0 0 1 1
**   1.000000000000000   1.000000000000000   0.0
**   1.024948365654583   0.934461926834452   0.0
**   1.116583018200151   0.777350964621867   0.0
**   1.295753434047077   0.606254343587194   0.0
**   1.537500000000000   0.462500000000000   0.0
**   1.768263070247418   0.329729152118310   0.0
**   1.920916981799849   0.185149035378133   0.0
**   1.986035130050921   0.054554577460044   0.0
**   2.000000000000000                 0.0   0.0
**                 0.0                 0.0   0.0
**   0.035513826946206   0.105291711848750   0.0
**   0.148591270347399   0.317731556850611   0.0
**   0.340010713990041   0.452219430075470   0.0
**   0.575000000000000   0.462500000000000   0.0
**   0.788022294598950   0.483764065630034   0.0
**   0.926408729652601   0.644768443149389   0.0
**   0.986453164464803   0.883724792445746   0.0
**   1.000000000000000   1.000000000000000   0.0
**  4 2 6 3
**  0 0 0 1
**   2.000000000000000                 0.0   0.0
**   1.986035130050921   0.054554577460044   0.0
**   1.920916981799849   0.185149035378133   0.0
**   1.768263070247418   0.329729152118310   0.0
**   1.537500000000000   0.462500000000000   0.0
**   1.295753434047077   0.606254343587194   0.0
**   1.116583018200151   0.777350964621867   0.0
**   1.024948365654583   0.934461926834452   0.0
**   1.000000000000000   1.000000000000000   0.0
**  7 2 4 1
**  0 0 0 0
**  Bezier --- Slant ---
**  --- Right --- Top
**  Bottom --- Right ---
```