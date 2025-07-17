# P4est-based mesh

The [`P4estMesh`](@ref) is an unstructured, curvilinear, nonconforming
mesh type for quadrilateral (2D) and hexahedral (3D) cells.
It supports quadtree/octree-based adaptive mesh refinement (AMR) via
the C library [`p4est`](https://github.com/cburstedde/p4est). See
[`AMRCallback`](@ref) for further information.

Due to its curvilinear nature, (numerical) fluxes need to implement methods
dispatching on the `normal::AbstractVector`. Rotationally invariant equations
such as the compressible Euler equations can use [`FluxRotated`](@ref) to
wrap numerical fluxes implemented only for Cartesian meshes. This simplifies
the re-use of existing functionality for the [`TreeMesh`](@ref) but is usually
less efficient, cf. [PR #550](https://github.com/trixi-framework/Trixi.jl/pull/550).

## Construction of a P4estMesh from an Abaqus file

One available option to construct a [`P4estMesh`](@ref) is to read in an Abaqus (`.inp`) mesh file.
We briefly describe the structure of this file, the conventions it uses, and how the mesh file
is parsed to create an initial unstructured, curvilinear, and conforming mesh.

### Mesh in two spatial dimensions

For this discussion we use the following two-dimensional unstructured curved mesh with three elements:

![abaqus-2dmesh-docs](https://user-images.githubusercontent.com/25242486/139241997-88e70a01-286f-4cee-80b1-2fd83c60bcca.png)

We note that the corner and element connectivity information parsed from the Abaqus file creates
a straight sided (linear) mesh.
From this linear mesh there are two strategies available to make the mesh curvilinear:

1. Apply a `mapping` function to describe a transformation of the linear mesh to another
   physical domain. The mapping is approximated using interpolation polynomial of a user
   specified polynomial degree. The default value of this polynomial degree is `1` that
   corresponds to an uncurved geometry.
2. High-order boundary information is available in the `.inp` mesh file because it was created
   with the [HOHQMesh](https://github.com/trixi-framework/HOHQMesh) mesh generator, which
   is available via the Julia package [HOHQMesh.jl](https://github.com/trixi-framework/HOHQMesh.jl).
   This information is used to create appropriate transfinite mappings during the mesh construction.

We divide our discussion into two parts. The first part discusses the standard corner and element information
contained in the `.inp` mesh file. The second part specifically deals with the mesh file parsing of an Abaqus
file created by HOHQMesh.jl.

#### Mesh file header

An Abaqus `.inp` mesh file typically begins with a `*Heading`.
Though *optional*, the `*Heading` is helpful to give users some information about the mesh described by the mesh file.
In particular, a `.inp` mesh file created with `HOHQMesh` will contain the header
```
*Heading
 File created by HOHQMesh
```
This heading is used to indicate to the mesh constructor which of the above mapping strategies to apply in order to
create a curvilinear mesh.
If the Abaqus file header is **not** present then the `P4estMesh` is created with the first strategy above.

#### [List of corner nodes](@id corner-node-list)

Next, prefaced with `*NODE`, comes a list of the physical `(x,y,z)` coordinates of all the corners.
The first integer in the list of the corners provides its id number.
Thus, for the two-dimensional example mesh this block of corner information is
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

#### [List of elements](@id element-list)

The element connectivity is given after the list of corners. The header for this information block is
```
*ELEMENT, type=CPS4, ELSET=Surface1
```
The Abaqus element type `CPS4` corresponds to a quadrilateral element.
Each quadrilateral element in the unstructured mesh is dictated by four corner points with indexing
taken from the numbering given by the corner list above.
The elements connect a set of four corner points (starting from the bottom left) in an anti-clockwise fashion;
making the element *right-handed*.
This element handedness is indicated using the circular arrow in the figure above.
Just as with the corner list, the first integer in the element connectivity list indicates the element id number.
Thus, the element connectivity list for the three element example mesh is
```
*ELEMENT, type=CPS4, ELSET=Surface1
1, 5, 1, 4, 3
2, 4, 2, 6, 3
3, 7, 2, 4, 1
```

#### Element neighbor connectivity

The construction of the element neighbor ids and identifying physical boundary surfaces is done using functionality
directly from the [`p4est`](https://github.com/cburstedde/p4est) library.
For example, the neighbor connectivity is created in the mesh constructor using the wrapper `read_inp_p4est` function.

#### Encoding of boundaries

##### HOHQMesh boundary information

If present, any additional information in the mesh file that was created by `HOHQMesh` is prefaced with
`** ` to make it an Abaqus comment.
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
The labels can be short descriptive words up to 32 characters in length.
The label `---` indicates an internal surface where no boundary condition is required.

It is important to note that these labels are given in the following order according to the
local surface index `-x` `+x` `-y` `+y` as required by the [`p4est`](https://github.com/cburstedde/p4est) library.
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

##### Standard Abaqus format boundary information

As an alternative to an Abaqus mesh generated by `HOHQMesh`, `.inp` files with boundary information encoded as nodesets `*NSET,NSET=` can be used to construct a `p4est` mesh.
This is especially useful for usage of existing meshes (consisting of bilinear elements) which could stem from the popular [`gmsh`](https://gmsh.info/) meshing software.

In addition to the list of [nodes](@ref corner-node-list) and [elements](@ref element-list) given above, there are nodesets of the form 
```
*NSET,NSET=PhysicalLine1
1, 4, 52, 53, 54, 55, 56, 57, 58, 
```
present which are used to associate the edges defined through their corner nodes with a label. In this case it is called `PhysicalLine1`.
By looping over every element and its associated edges, consisting of two nodes, we query the read in `NSET`s if the current node pair is present.

To prevent that every nodeset following `*NSET,NSET=` is treated as a boundary, the user must supply a `boundary_symbols` keyword to the [`P4estMesh`](@ref) constructor:

```julia
boundary_symbols = [:PhysicalLine1]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)
```
By doing so, only nodesets with a label present in `boundary_symbols` are treated as physical boundaries.
Other nodesets that could be used for diagnostics are not treated as external boundaries.
Note that there is a leading colon `:` compared to the label in the `.inp` mesh file.
This is required to turn the label into a [`Symbol`](https://docs.julialang.org/en/v1/manual/metaprogramming/#Symbols).
**Important**: In Julia, a symbol _cannot_ contain a hyphen/dash `-`, i.e., `:BC-1` is _not_ a valid symbol.
Keep this in mind when importing boundaries, you might have to convert hyphens/dashes `-` to underscores `_` in the `.inp` mesh file, i.e., `BC_1` instead of `BC-1`.

A 2D example for this mesh, which is read-in for an unstructured mesh file created with `gmsh`, is presented in 
`examples/p4est_2d_dgsem/elixir_euler_NACA6412airfoil_mach2.jl`.

### Mesh in three spatial dimensions

#### `HOHQMesh`-Extended Abaqus format

The 3D Abaqus file format with high-order boundary information from `HOHQMesh` is very similar to the
2D version discussed above. There are only three changes:

1. The element connectivity would be given in terms of the eight corners that define a hexahedron.
   The corners are numbered as shown in the figure below. The header of the element list changes to be
   ```
   *ELEMENT, type=C3D8, ELSET=Volume1
   ```
   where `C3D8` corresponds to a Abaqus hexahedral element.
2. There are six check digits included directly below the eight corner indexes to indicate whether
   the local face within the element is straight sided, `0`, or is curved, `1`. For curved faces
   `(x,y,z)` coordinate values are available in order to construct an face interpolant with the mesh
   polynomial order at the Chebyshev-Gauss-Lobatto nodes.
3. The boundary labels are given in the following order according to the local surface index
   `-x` `+x` `-y` `+y` `-z` `+z` as required by the [`p4est`](https://github.com/cburstedde/p4est) library.

For completeness, we also give a short description and derivation of the three-dimensional transfinite mapping
formulas used to compute the physical coordinates $\mathbf{x}=(x,y,z)$ of a (possibly curved) hexahedral element
give the reference coordinates $\boldsymbol{\xi} = (\xi, \eta, \zeta)$ which lie in $[-1,1]^3$. That is, we will
create an expression $\mathbf{x}= \mathbf{X}(\boldsymbol{\xi})$.

Below we provide a sketch of a single hexahedral element with curved faces. This is done to introduce the numbering
conventions for corners, edges, and faces of the element.

![abaqus-3dmesh-docs](https://user-images.githubusercontent.com/25242486/139839161-8c5f5979-2724-4cfb-9eac-6af58105ef12.png)

When the hexahedron is a straight sided (linear) element we compute the transfinite mapping directly from the
element corner points according to
```math
\begin{aligned}
\mathbf{X}_{linear}(\boldsymbol{\xi}) &=  \frac{1}{8}[\quad\, \mathbf{x}_1(1-\xi)(1-\eta)(1-\zeta)
                                                         + \mathbf{x}_2(1+\xi)(1-\eta)(1-\zeta)\\[-0.15cm]
                                    & \qquad\;             + \mathbf{x}_3(1+\xi)(1+\eta)(1-\zeta)
                                                         + \mathbf{x}_4(1-\xi)(1+\eta)(1-\zeta) \\
                                    & \qquad\;             + \mathbf{x}_5(1-\xi)(1-\eta)(1+\zeta)
                                                         + \mathbf{x}_6(1+\xi)(1-\eta)(1+\zeta) \\
                                    & \qquad\;             + \mathbf{x}_7(1+\xi)(1+\eta)(1+\zeta)
                                                         + \mathbf{x}_8(1-\xi)(1+\eta)(1+\zeta)\quad].
\end{aligned}
```

Next, we create a transfinite mapping function, $\mathbf{X}(\boldsymbol{\xi})$, for a hexahedron that
has one or more curved faces. For this we assume that have a set of six interpolating polynomials
$\{\Gamma_i\}_{i=1}^6$ that approximate the faces. The interpolating polynomial for any curved faces is provided
by the information in a `HOHQMesh` Abaqus mesh file or is constructed on the fly via a
bi-linear interpolation routine for any linear faces. Explicitly, these six face interpolation polynomials depend
on the computational coordinates $\boldsymbol{\xi}$ as follows
```math
  \begin{aligned}
    \Gamma_1(\xi, \zeta), \quad && \quad \Gamma_3(\xi, \eta), \quad && \quad \Gamma_4(\eta, \zeta),\\[0.1cm]
    \Gamma_2(\xi, \zeta), \quad && \quad \Gamma_5(\xi, \eta), \quad && \quad \Gamma_6(\eta, \zeta).
  \end{aligned}
```

To determine the form of the mapping we first create linear interpolations between two opposing faces, e.g., $\Gamma_3$ and $\Gamma_5$ and sum them together to have
```math
\begin{aligned}
  \boldsymbol\Sigma(\boldsymbol{\xi}) &= \frac{1}{2}[\quad\,(1-\xi)\Gamma_6(\eta,\zeta) + (1+\xi)\Gamma_4(\eta,\zeta) \\[-0.15cm]
  &\qquad\;+ (1-\eta)\Gamma_1(\xi,\zeta) + (1+\eta)\Gamma_2(\xi,\zeta) \\%[-0.15cm]
                                  &\qquad\; +(1-\zeta)\Gamma_3(\xi,\eta) + (1+\zeta)\Gamma_5(\xi,\eta)\quad].
\end{aligned}
```

Unfortunately, the linear interpolations $\boldsymbol\Sigma(\boldsymbol{\xi})$ no longer match at the faces, e.g., evaluating at $\eta = -1$ we have
```math
\boldsymbol\Sigma(\xi,-1,\zeta) = \Gamma_1(\xi,\zeta) + \frac{1}{2}[\;(1-\xi)\Gamma_6(-1,\zeta) + (1+\xi)\Gamma_4(-1,\zeta)
                                 +(1-\zeta)\Gamma_3(\xi,-1) + (1+\zeta)\Gamma_5(\xi,-1)\;],
```
which is the desired face $\Gamma_1(\xi,\zeta)$ plus four edge error terms.
Analogous edge error terms occur at the other faces evaluating $\boldsymbol\Sigma(\boldsymbol{\xi})$
at $\eta=1$, $\xi=\pm 1$, and $\zeta=\pm 1$.
In order to match the faces, we subtract a linear interpolant in the $\xi$, $\eta$, and $\zeta$ directions of the
edge error terms, e.g., the terms in braces in the above equation. So, continuing the example above, the correction term to be subtracted for face $\Gamma_1$ to match would be
```math
\left(\frac{1-\eta}{2}\right) \bigg[ \frac{1}{2} [ \; (1-\xi)\Gamma_6(-1,\zeta) + (1+\xi)\Gamma_4(-1,\zeta)+(1-\zeta)\Gamma_3(\xi,-1)
 + (1+\zeta)\Gamma_5(\xi,-1)\;] \bigg].
```
For clarity, and to allow an easier comparison to the implementation, we introduce auxiliary notation for the 12 edge
values present in the complete correction term. That is, for given values of $\xi$, $\eta$, and $\zeta$ we have
```math
  \begin{aligned}
    \texttt{edge}_{1} &= \Gamma_1(\xi, -1), \quad && \quad \texttt{edge}_{5} = \Gamma_2(\xi, -1), \quad & \quad  \texttt{edge}_{9} &= \Gamma_6(\eta, -1),\\[0.1cm]
    \texttt{edge}_{2} &= \Gamma_1(1, \zeta), \quad && \quad\texttt{edge}_{6} = \Gamma_2(1, \zeta), \quad & \quad  \texttt{edge}_{10} &= \Gamma_4(\eta, -1),\\[0.1cm]
    \texttt{edge}_{3} &= \Gamma_1(\xi, 1), \quad && \quad \texttt{edge}_{7} = \Gamma_2(\xi,  1), \quad & \quad  \texttt{edge}_{11} &= \Gamma_4(\eta, 1),\\[0.1cm]
    \texttt{edge}_{4} &= \Gamma_1(-1, \zeta), \quad && \quad \texttt{edge}_{8} = \Gamma_2(-1, \zeta), \quad & \quad  \texttt{edge}_{12} &= \Gamma_6(\eta, 1).
  \end{aligned}
```
With this notation for the edge terms (and after some algebraic manipulation) we write the complete edge correction term,
$\mathcal{C}_{\texttt{edge}}(\boldsymbol{\xi})$, as
```math
\begin{aligned}
\mathcal{C}_{\texttt{edge}}(\boldsymbol{\xi}) &=  \frac{1}{4}[\quad\, (1-\eta)(1-\zeta)\texttt{edge}_{1}\\[-0.15cm]
                                    & \qquad\;              + (1+\xi)(1-\eta)\texttt{edge}_{2} \\
                                    & \qquad\;              + (1-\eta)(1+\zeta)\texttt{edge}_{3} \\
                                    & \qquad\;              + (1-\xi)(1-\eta)\texttt{edge}_{4} \\
                                    & \qquad\;              + (1+\eta)(1-\zeta)\texttt{edge}_{5} \\
                                    & \qquad\;              + (1+\xi)(1+\eta)\texttt{edge}_{6} \\
                                    & \qquad\;              + (1+\eta)(1+\zeta)\texttt{edge}_{7} \\
                                    & \qquad\;              + (1-\xi)(1+\eta)\texttt{edge}_{8} \\
                                    & \qquad\;              + (1-\xi)(1-\zeta)\texttt{edge}_{9} \\
                                    & \qquad\;              + (1+\xi)(1-\zeta)\texttt{edge}_{10} \\
                                    & \qquad\;              + (1+\xi)(1+\zeta)\texttt{edge}_{11} \\
                                    & \qquad\;              + (1-\xi)(1+\zeta)\texttt{edge}_{12}\quad].
\end{aligned}
```

However, subtracting the edge correction terms $\mathcal{C}_{\texttt{edge}}(\boldsymbol{\xi})$
from $\boldsymbol\Sigma(\boldsymbol{\xi})$ removes the interior element contributions twice.
Thus, to complete the construction of the transfinite mapping $\mathbf{X}(\boldsymbol{\xi})$ we add the
transfinite map of the straight sided hexahedral element to find
```math
\mathbf{X}(\boldsymbol{\xi}) = \boldsymbol\Sigma(\boldsymbol{\xi})
                             - \mathcal{C}_{\texttt{edge}}(\boldsymbol{\xi})
                             + \mathbf{X}_{linear}(\boldsymbol{\xi}).
```

#### Construction from standard Abaqus

Also for a mesh in standard Abaqus format there are no qualitative changes when going from 2D to 3D.
The most notable difference is that boundaries are formed in 3D by faces defined by four nodes while in 2D boundaries are edges consisting of two elements.
Note that standard Abaqus also defines quadratic element types. In Trixi.jl, these higher-order elements are currently only supported in 2D, i.e., 8-node quadrilaterals.
A simple mesh file, which is used also in `examples/p4est_3d_dgsem/elixir_euler_free_stream_boundaries.jl`, is given below:
```
*Heading
### SOMETHING DIFFERENT FROM "File created by HOHQMesh" ###
*NODE
1, -2, 0, 0
2, -1, 0, 0
3, -1, 1, 0
4, -2, 1, 0
5, -2, 0, 1
6, -1, 0, 1
7, -1, 1, 1
8, -2, 1, 1
9, -1.75, 1, 0
10, -1.5, 1, 0
11, -1.25, 1, 0
12, -1, 0.75000000000035, 0
13, -1, 0.50000000000206, 0
14, -1, 0.25000000000104, 0
15, -1.25, 0, 0
16, -1.5, 0, 0
17, -1.75, 0, 0
18, -2, 0.24999999999941, 0
19, -2, 0.49999999999869, 0
20, -2, 0.74999999999934, 0
21, -1.75, 0, 1
22, -1.5, 0, 1
23, -1.25, 0, 1
24, -1, 0.24999999999941, 1
25, -1, 0.49999999999869, 1
26, -1, 0.74999999999934, 1
27, -1.25, 1, 1
28, -1.5, 1, 1
29, -1.75, 1, 1
30, -2, 0.75000000000035, 1
31, -2, 0.50000000000206, 1
32, -2, 0.25000000000104, 1
33, -2, 0, 0.24999999999941
34, -2, 0, 0.49999999999869
35, -2, 0, 0.74999999999934
36, -2, 1, 0.24999999999941
37, -2, 1, 0.49999999999869
38, -2, 1, 0.74999999999934
39, -1, 0, 0.24999999999941
40, -1, 0, 0.49999999999869
41, -1, 0, 0.74999999999934
42, -1, 1, 0.24999999999941
43, -1, 1, 0.49999999999869
44, -1, 1, 0.74999999999934
45, -1.25, 0.25000000000063, 0
46, -1.25, 0.50000000000122, 0
47, -1.25, 0.7500000000001, 0
48, -1.5, 0.25000000000023, 0
49, -1.5, 0.50000000000038, 0
50, -1.5, 0.74999999999984, 0
51, -1.75, 0.24999999999982, 0
52, -1.75, 0.49999999999953, 0
53, -1.75, 0.74999999999959, 0
54, -1.75, 0.25000000000063, 1
55, -1.75, 0.50000000000122, 1
56, -1.75, 0.7500000000001, 1
57, -1.5, 0.25000000000023, 1
58, -1.5, 0.50000000000038, 1
59, -1.5, 0.74999999999984, 1
60, -1.25, 0.24999999999982, 1
61, -1.25, 0.49999999999953, 1
62, -1.25, 0.74999999999959, 1
63, -2, 0.24999999999982, 0.24999999999941
64, -2, 0.49999999999953, 0.24999999999941
65, -2, 0.74999999999959, 0.24999999999941
66, -2, 0.25000000000023, 0.49999999999869
67, -2, 0.50000000000038, 0.49999999999869
68, -2, 0.74999999999984, 0.49999999999869
69, -2, 0.25000000000063, 0.74999999999934
70, -2, 0.50000000000122, 0.74999999999934
71, -2, 0.7500000000001, 0.74999999999934
72, -1.25, 1, 0.74999999999934
73, -1.25, 1, 0.49999999999869
74, -1.25, 1, 0.24999999999941
75, -1.5, 1, 0.74999999999934
76, -1.5, 1, 0.49999999999869
77, -1.5, 1, 0.24999999999941
78, -1.75, 1, 0.74999999999934
79, -1.75, 1, 0.49999999999869
80, -1.75, 1, 0.24999999999941
81, -1, 0.25000000000063, 0.24999999999941
82, -1, 0.50000000000122, 0.24999999999941
83, -1, 0.7500000000001, 0.24999999999941
84, -1, 0.25000000000023, 0.49999999999869
85, -1, 0.50000000000038, 0.49999999999869
86, -1, 0.74999999999984, 0.49999999999869
87, -1, 0.24999999999982, 0.74999999999934
88, -1, 0.49999999999953, 0.74999999999934
89, -1, 0.74999999999959, 0.74999999999934
90, -1.75, 0, 0.74999999999934
91, -1.75, 0, 0.49999999999869
92, -1.75, 0, 0.24999999999941
93, -1.5, 0, 0.74999999999934
94, -1.5, 0, 0.49999999999869
95, -1.5, 0, 0.24999999999941
96, -1.25, 0, 0.74999999999934
97, -1.25, 0, 0.49999999999869
98, -1.25, 0, 0.24999999999941
99, -1.75, 0.25000000000043, 0.74999999999934
100, -1.75, 0.25000000000023, 0.49999999999869
101, -1.75, 0.25000000000002, 0.24999999999941
102, -1.75, 0.5000000000008, 0.74999999999934
103, -1.75, 0.50000000000038, 0.49999999999869
104, -1.75, 0.49999999999995, 0.24999999999941
105, -1.75, 0.74999999999997, 0.74999999999934
106, -1.75, 0.74999999999984, 0.49999999999869
107, -1.75, 0.74999999999972, 0.24999999999941
108, -1.5, 0.25000000000023, 0.74999999999934
109, -1.5, 0.25000000000023, 0.49999999999869
110, -1.5, 0.25000000000023, 0.24999999999941
111, -1.5, 0.50000000000038, 0.74999999999934
112, -1.5, 0.50000000000038, 0.49999999999869
113, -1.5, 0.50000000000038, 0.24999999999941
114, -1.5, 0.74999999999984, 0.74999999999934
115, -1.5, 0.74999999999984, 0.49999999999869
116, -1.5, 0.74999999999984, 0.24999999999941
117, -1.25, 0.25000000000002, 0.74999999999934
118, -1.25, 0.25000000000023, 0.49999999999869
119, -1.25, 0.25000000000043, 0.24999999999941
120, -1.25, 0.49999999999995, 0.74999999999934
121, -1.25, 0.50000000000038, 0.49999999999869
122, -1.25, 0.5000000000008, 0.24999999999941
123, -1.25, 0.74999999999972, 0.74999999999934
124, -1.25, 0.74999999999984, 0.49999999999869
125, -1.25, 0.74999999999997, 0.24999999999941
******* E L E M E N T S *************
*ELEMENT, type=C3D8, ELSET=Volume1
153, 54, 21, 5, 32, 99, 90, 35, 69
154, 99, 90, 35, 69, 100, 91, 34, 66
155, 100, 91, 34, 66, 101, 92, 33, 63
156, 101, 92, 33, 63, 51, 17, 1, 18
157, 55, 54, 32, 31, 102, 99, 69, 70
158, 102, 99, 69, 70, 103, 100, 66, 67
159, 103, 100, 66, 67, 104, 101, 63, 64
160, 104, 101, 63, 64, 52, 51, 18, 19
161, 56, 55, 31, 30, 105, 102, 70, 71
162, 105, 102, 70, 71, 106, 103, 67, 68
163, 106, 103, 67, 68, 107, 104, 64, 65
164, 107, 104, 64, 65, 53, 52, 19, 20
165, 29, 56, 30, 8, 78, 105, 71, 38
166, 78, 105, 71, 38, 79, 106, 68, 37
167, 79, 106, 68, 37, 80, 107, 65, 36
168, 80, 107, 65, 36, 9, 53, 20, 4
169, 57, 22, 21, 54, 108, 93, 90, 99
170, 108, 93, 90, 99, 109, 94, 91, 100
171, 109, 94, 91, 100, 110, 95, 92, 101
172, 110, 95, 92, 101, 48, 16, 17, 51
173, 58, 57, 54, 55, 111, 108, 99, 102
174, 111, 108, 99, 102, 112, 109, 100, 103
175, 112, 109, 100, 103, 113, 110, 101, 104
176, 113, 110, 101, 104, 49, 48, 51, 52
177, 59, 58, 55, 56, 114, 111, 102, 105
178, 114, 111, 102, 105, 115, 112, 103, 106
179, 115, 112, 103, 106, 116, 113, 104, 107
180, 116, 113, 104, 107, 50, 49, 52, 53
181, 28, 59, 56, 29, 75, 114, 105, 78
182, 75, 114, 105, 78, 76, 115, 106, 79
183, 76, 115, 106, 79, 77, 116, 107, 80
184, 77, 116, 107, 80, 10, 50, 53, 9
185, 60, 23, 22, 57, 117, 96, 93, 108
186, 117, 96, 93, 108, 118, 97, 94, 109
187, 118, 97, 94, 109, 119, 98, 95, 110
188, 119, 98, 95, 110, 45, 15, 16, 48
189, 61, 60, 57, 58, 120, 117, 108, 111
190, 120, 117, 108, 111, 121, 118, 109, 112
191, 121, 118, 109, 112, 122, 119, 110, 113
192, 122, 119, 110, 113, 46, 45, 48, 49
193, 62, 61, 58, 59, 123, 120, 111, 114
194, 123, 120, 111, 114, 124, 121, 112, 115
195, 124, 121, 112, 115, 125, 122, 113, 116
196, 125, 122, 113, 116, 47, 46, 49, 50
197, 27, 62, 59, 28, 72, 123, 114, 75
198, 72, 123, 114, 75, 73, 124, 115, 76
199, 73, 124, 115, 76, 74, 125, 116, 77
200, 74, 125, 116, 77, 11, 47, 50, 10
201, 24, 6, 23, 60, 87, 41, 96, 117
202, 87, 41, 96, 117, 84, 40, 97, 118
203, 84, 40, 97, 118, 81, 39, 98, 119
204, 81, 39, 98, 119, 14, 2, 15, 45
205, 25, 24, 60, 61, 88, 87, 117, 120
206, 88, 87, 117, 120, 85, 84, 118, 121
207, 85, 84, 118, 121, 82, 81, 119, 122
208, 82, 81, 119, 122, 13, 14, 45, 46
209, 26, 25, 61, 62, 89, 88, 120, 123
210, 89, 88, 120, 123, 86, 85, 121, 124
211, 86, 85, 121, 124, 83, 82, 122, 125
212, 83, 82, 122, 125, 12, 13, 46, 47
213, 7, 26, 62, 27, 44, 89, 123, 72
214, 44, 89, 123, 72, 43, 86, 124, 73
215, 43, 86, 124, 73, 42, 83, 125, 74
216, 42, 83, 125, 74, 3, 12, 47, 11
*NSET,NSET=PhysicalSurface1
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
31, 32, 33, 34, 35, 36, 37, 38, 45, 46, 
47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 
67, 68, 69, 70, 71, 
*NSET,NSET=PhysicalSurface2
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 
24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 
37, 38, 39, 40, 41, 42, 43, 44, 72, 73, 
74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 
94, 95, 96, 97, 98, 
```

## Interfacing with the `p4est` mesh backend

Sometimes it is necessary to exchange data between the solver and the mesh.
In the case of the [`P4estMesh`](@ref) and its [`p4est`](https://www.p4est.org/), this is more intricate than with other mesh types, since the `p4est` backend
is comparatively opaque to the rest of the code due to being a compiled C library.

As a disclaimer: This is neither complete nor self-contained, and is meant as a first orientation guide.
To implement own features, you likely need to take a look at the documentation of [`p4est`](https://github.com/cburstedde/p4est), its Julia wrapper [P4est.jl](https://github.com/trixi-framework/P4est.jl), and relevant parts of the [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) source code.
Additionally, the [Julia manual for interfacing C code](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/) and the [documentation of the Julia C interface](https://docs.julialang.org/en/v1/base/c/) are also helpful.

First, we go through the implementation of adaptive mesh refinement (AMR) with `p4est`.
Second, we show how a custom mesh partitioning based on a weighting function can be implemented.

### Adaptive Mesh Refinement (AMR)

In a simulation that uses adaptive mesh refinement (AMR), cells are assigned an indicator value $I \in \{-1, 0, 1\}$ that denotes whether the cell should be coarsened ($I = -1$), kept at the current level ($I = 0$), or be refined ($I = 1$).

For realizing AMR there are now in principle two possibilities:
- First, the solver could receive the connectivity information from the mesh backend and then handle the refinement and coarsening of the mesh.
- Second, the solver could send the indicator values to the mesh backend and let it handle the refinement and coarsening of the mesh.

For the `p4est` mesh, Trixi.jl uses the latter approach to benefit from the highly efficient implementation of the `p4est` library.

The first step involves sending the indicator values to the mesh backend.
The indicator values [`lambda` are obtained from a "controller" in the `AMRCallback`](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/callbacks_step/amr.jl#L657-L658)
```julia
lambda = @trixi_timeit timer() "indicator" controller(u, mesh, equations, dg, cache,
                                                      t = t, iter = iter)
```
and need now to be sent to the mesh backend.
To this end, a number of helper functions are required.
First, [a Julia function with two arguments `info, user_data` is defined that copies the indicator values from `user_data` to a field in `info`](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/callbacks_step/amr.jl#L517-L539).
The `info` is a field from the `p4est` mesh (precisely `p4est_iter_volume_info_t` - for details see the [`p4est` documentation](https://p4est.github.io/api/p4est-latest/p4est__iterate_8h.html)):
```julia
# Copy controller values to p4est quad user data storage
function copy_to_quad_iter_volume(info, user_data)
    info_pw = PointerWrapper(info)

    # Load tree from global trees array, one-based indexing
    tree_pw = load_pointerwrapper_tree(info_pw.p4est, info_pw.treeid[] + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree_pw.quadrants_offset[]
    # Global quad ID
    quad_id = offset + info_pw.quadid[]

    # Access user_data = `lambda`
    user_data_pw = PointerWrapper(Int, user_data)
    # Load controller_value = `lambda[quad_id + 1]`
    controller_value = user_data_pw[quad_id + 1]

    # Access quadrant's user data `[global quad ID, controller_value]`
    quad_data_pw = PointerWrapper(Int, info_pw.quad.p.user_data[])
    # Save controller value to quadrant's user data.
    quad_data_pw[2] = controller_value

    return nothing
end
```
This function is then converted to a C function to match the required syntax of the [`p4est_iter_volume_t`](https://github.com/cburstedde/p4est/blob/2296a990d8b6b54731a63be0ba5bc17b08cd1f3d/src/p4est_iterate.h#L57-L63)/[`p8est_iter_volume_t`](https://github.com/cburstedde/p4est/blob/2296a990d8b6b54731a63be0ba5bc17b08cd1f3d/src/p8est_iterate.h#L57-L63) function:
```C
/** The prototype for a function that p4est_iterate will execute at every
 * quadrant local to the current process.
 * \param [in] info          information about a quadrant provided to the user
 * \param [in,out] user_data the user context passed to p4est_iterate()
 */
typedef void (*p4est_iter_volume_t) (p4est_iter_volume_info_t * info,
                                     void *user_data);

/** The prototype for a function that p8est_iterate() will execute at every
 * quadrant local to the current process.
 * \param [in] info          information about a quadrant provided to the user
 * \param [in,out] user_data the user context passed to p8est_iterate()
 */
typedef void (*p8est_iter_volume_t) (p8est_iter_volume_info_t * info,
                                     void *user_data);
```
For these functions documentation is provided for both [2D](https://p4est.github.io/api/p4est-latest/p4est__iterate_8h.html) and [3D](https://p4est.github.io/api/p4est-latest/p8est__iterate_8h.html).

To handle both 2D (which is the canonical `p4est`) and 3D (which is within the `p4est` library referred to as `p8est`), we need to construct [dispatching functions](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/callbacks_step/amr.jl#L638-L647). The dispatching is performed on the number of spatial dimensions:
```julia
# 2D
function cfunction(::typeof(copy_to_quad_iter_volume), ::Val{2})
    @cfunction(copy_to_quad_iter_volume, 
               # Note the matching signature to the C function: 
               Cvoid, # Return type
               # Parameters
               (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(copy_to_quad_iter_volume), ::Val{3})
    @cfunction(copy_to_quad_iter_volume, 
               # Note the matching signature to the C function
               Cvoid, # Return type
               # Parameters
               (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))
end
```

The correct function is then [selected via](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/callbacks_step/amr.jl#L558-L559)
```julia
iter_volume_c = cfunction(copy_to_quad_iter_volume, Val(ndims(mesh)))
```
and [by calling `iterate_p4est` the indicator values are finally copied to the cells](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/callbacks_step/amr.jl#L667-L669) (quadrants or octants) of the mesh:
```julia
iterate_p4est(mesh.p4est, # The p4est mesh
              lambda; # The indicator values
              # The function that copies the indicator values to the mesh
              iter_volume_c = iter_volume_c)
```

Now, during the `refine!` or `coarsen!` calls (which eventually call [`refine_fn`](https://p4est.github.io/api/p4est-latest/p4est_8h.html#a1a31375edfa42b5609e7656233c32cca)/[`coarsen_fn`](https://p4est.github.io/api/p4est-latest/p4est_8h.html#ad250f4765d9778ec3940e9fabea7c853) from `p4est`) the indicator values are available in the user data of the mesh cells.
In particular, again a function needs to be provided that matches the signature of the `p4est_refine_t` [[Documentation](https://p4est.github.io/api/p4est-latest/p4est_8h.html#ad6f6d433abde78f20ea267e6aebea26a), [Implementation](https://github.com/cburstedde/p4est/blob/2296a990d8b6b54731a63be0ba5bc17b08cd1f3d/src/p4est.h#L221-L229)]/`p8est_refine_t`[[Documentation](https://p4est.github.io/api/p4est-latest/p8est_8h.html#a24565b65860e156a04ba8ccc6f67a936), [Implementation](https://github.com/cburstedde/p4est/blob/2296a990d8b6b54731a63be0ba5bc17b08cd1f3d/src/p8est.h#L217-L225)] function
```C
/** Callback function prototype to decide for refinement.
 * \param [in] p4est       the forest
 * \param [in] which_tree  the tree containing \a quadrant
 * \param [in] quadrant    the quadrant that may be refined
 * \return nonzero if the quadrant shall be refined.
 */
typedef int (*p4est_refine_t) (p4est_t * p4est,
                               p4est_topidx_t which_tree,
                               p4est_quadrant_t * quadrant);

/** Callback function prototype to decide for refinement.
 * \param [in] p8est       the forest
 * \param [in] which_tree  the tree containing \a quadrant
 * \param [in] quadrant    the quadrant that may be refined
 * \return nonzero if the quadrant shall be refined.
 */
typedef int (*p8est_refine_t) (p8est_t * p8est,
                               p4est_topidx_t which_tree,
                               p8est_quadrant_t * quadrant);
```
which [is implemented as](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/meshes/p4est_mesh.jl#L2227-L2242)
```julia
function refine_fn(p4est, which_tree, quadrant)
    # Controller value has been copied to the quadrant's user data storage before.
    # Unpack quadrant's user data ([global quad ID, controller_value]).
    # Use `unsafe_load` here since `quadrant.p.user_data isa Ptr{Ptr{Nothing}}`
    # and we only need the first (only!) entry
    pw = PointerWrapper(Int, unsafe_load(quadrant.p.user_data))
    controller_value = pw[2]

    if controller_value > 0
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end
```
Again, [dimensional dispatching](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/meshes/p4est_mesh.jl#L2244-L2253) is used to handle both 2D and 3D meshes:
```julia
# 2D
function cfunction(::typeof(refine_fn), ::Val{2})
    @cfunction(refine_fn, 
               # Note the matching signature to the C function
               Cint, # Return type
               # Parameters
               (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))
end
# 3D
function cfunction(::typeof(refine_fn), ::Val{3})
    @cfunction(refine_fn, 
               # Note the matching signature to the C function
               Cint, # Return type
               # Parameters
               (Ptr{p8est_t}, Ptr{p4est_topidx_t}, Ptr{p8est_quadrant_t}))
end
```
These are then handed over to [`refine_p4est!`](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/auxiliary/p4est.jl#L144-L154), which eventually calls `refine_fn`.

### Custom mesh partitioning

Analogous to the AMR example above, we now discuss how to implement a custom mesh partitioning based on custom cell weighting.
This might be required for distributed memory (i.e., MPI-parallelized simulations) to achieve a more uniform distribution of the computational work across the different processes.
This is important to maintain scalability and to avoid load imbalance.

We begin again by copying the solver data.
Here we assume that the weighting should be done based on the user data `rhs_per_element`, which arises for instance in multirate/local time-stepping solvers where the computational work per cell is no longer uniform, but can be estimated by the number of right-hand-side (RHS) evaluations per cell.
Thus, a function, which saves this user data to the mesh, could look like this:
```julia
function save_rhs_evals_iter_volume(info, user_data)
    info_pw = PointerWrapper(info)

    # Load tree from global trees array, one-based indexing
    tree_pw = load_pointerwrapper_tree(info_pw.p4est, info_pw.treeid[] + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree_pw.quadrants_offset[]
    # Global quad ID
    quad_id = offset + info_pw.quadid[]

    # Access user_data = `rhs_per_element`
    user_data_pw = PointerWrapper(Int, user_data)
    # Load `rhs_evals = rhs_per_element[quad_id + 1]`
    rhs_evals = user_data_pw[quad_id + 1]

    # Access quadrant's user data (`[rhs_evals]`)
    quad_data_pw = PointerWrapper(Int, info_pw.quad.p.user_data[])
    # Save number of rhs evaluations to quadrant's user data.
    quad_data_pw[1] = rhs_evals

    return nothing
end
```
Again, dimensional dispatching is used to handle both 2D and 3D meshes:
```julia
# 2D
function cfunction(::typeof(save_rhs_evals_iter_volume), ::Val{2})
    @cfunction(save_rhs_evals_iter_volume, 
               # Note the matching signature to the C function
               Cvoid, # Return type
               # Parameters
               (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(save_rhs_evals_iter_volume), ::Val{3})
    @cfunction(save_rhs_evals_iter_volume, 
               # Note the matching signature to the C function
               Cvoid, # Return type
               # Parameters
               (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))
end
```

Similar to the `refine_fn` function, `p4est` requires a weighting function with the same signature [[Documentation](https://p4est.github.io/api/p4est-latest/p4est_8h.html#aa03358f1326e23d122ef1b155705fd4d), [Implementation](https://github.com/cburstedde/p4est/blob/2296a990d8b6b54731a63be0ba5bc17b08cd1f3d/src/p4est.h#L241-L249)]:
```C
/** Callback function prototype to calculate weights for partitioning.
 * \param [in] p4est       the forest
 * \param [in] which_tree  the tree containing \a quadrant
 * \return a 32bit integer >= 0 as the quadrant weight.
 * \note    Global sum of weights must fit into a 64bit integer.
 */
typedef int (*p4est_weight_t) (p4est_t * p4est,
                               p4est_topidx_t which_tree,
                               p4est_quadrant_t * quadrant);
```
Again, the 3D version [[Documentation](https://p4est.github.io/api/p4est-latest/p8est_8h.html#a065466172704df28878d8535b98965a1), [Implementation](https://github.com/cburstedde/p4est/blob/2296a990d8b6b54731a63be0ba5bc17b08cd1f3d/src/p8est.h#L237-L245)] is very similar:
```C
/** Callback function prototype to calculate weights for partitioning.
 * \param [in] p8est       the forest
 * \param [in] which_tree  the tree containing \a quadrant
 * \return a 32bit integer >= 0 as the quadrant weight.
 * \note    Global sum of weights must fit into a 64bit integer.
 */
typedef int (*p8est_weight_t) (p8est_t * p8est,
                               p4est_topidx_t which_tree,
                               p8est_quadrant_t * quadrant);
```

The equivalent Julia function is then very simply implemented as
```julia
function weight_fn_multirate(p4est, which_tree, quadrant)
    # Number of RHS evaluations has been copied to the quadrant's user data storage before.
    # Unpack quadrant's user data ([rhs_evals]).
    # Use `unsafe_load` here since `quadrant.p.user_data isa Ptr{Ptr{Nothing}}`
    # and we only need the first entry
    pw = PointerWrapper(Int, unsafe_load(quadrant.p.user_data))
    weight = pw[1] # rhs_evals

    return Cint(weight)
end

# Dimensional dispatch
# 2D
function cfunction(::typeof(weight_fn_multirate), ::Val{2})
    @cfunction(weight_fn_multirate, 
               # Note the matching signature to the C function
               Cint, # Return type
               # Parameters
               (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))
end

# 3D
function cfunction(::typeof(weight_fn_multirate), ::Val{3})
    @cfunction(weight_fn_multirate,
               # Note the matching signature to the C function
               Cint, # Return type
               # Parameters
               (Ptr{p8est_t}, Ptr{p4est_topidx_t}, Ptr{p8est_quadrant_t}))
end
```

Then, a different balance function could look like this:
```julia
function balance_p4est_multirate!(mesh::ParallelP4estMesh, dg, cache,
                                  # Vector{Vector{Int}} = Multirate levels with associated element indices
                                  level_info_elements, 
                                  # Vector{Int} = RHS Runge-Kutta stage evaluations per multirate level
                                  stages)
    rhs_per_element = zeros(Int, nelements(dg, cache))

    # Loop over different multirate levels
    for level in eachindex(level_info_elements)
        # Access computational cost = RHS evaluations for this multirate level
        rhs_evals = stages[level]

        # Loop over all elements on this multirate level
        for element in level_info_elements[level]
            rhs_per_element[element] = rhs_evals
        end
    end

    # Copy `rhs_per_element` to the `p4est` mesh backend
    iter_volume_c = cfunction(save_rhs_evals_iter_volume, Val(ndims(mesh)))
    iterate_p4est(mesh.p4est, rhs_per_element; iter_volume_c = iter_volume_c)

    # Set up according weighting function that can now be evaluated since
    # the user data has been copied to the mesh backend
    weight_fn_c = cfunction(weight_fn_multirate, Val(ndims(mesh)))
    # Perform mesh partitioning with custom weighting function
    partition!(mesh; weight_fn = weight_fn_c)
end
```
This redistributes the mesh among the MPI ranks.
Now, the same needs to be done for the solver (recall that mesh and solver are quite decoupled for `p4est` meshes).
Thus, the call to [`partition!`](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/meshes/p4est_mesh.jl#L2189-L2197) needs to be followed by a call to [`rebalance_solver!`](https://github.com/trixi-framework/Trixi.jl/blob/f76ccaf23a1df150f9a185f0faa770ebc1abc417/src/callbacks_step/amr_dg.jl#L8-L92) to rebalance the solver data.
Overall, this non-standard partitioning can be realized as
```julia
# Get cell distribution for standard partitioning
global_first_quadrant = unsafe_wrap(Array,
                                    unsafe_load(mesh.p4est).global_first_quadrant,
                                    mpi_nranks() + 1)
# Need to copy `global_first_quadrant` to different variable as the former will change 
# due to the call to `partition!`
old_global_first_quadrant = copy(global_first_quadrant)

# Get (global) element distribution to accordingly balance the solver
level_info_elements = some_user_defined_function()

# Balance such that each rank has the same number of RHS calls                                    
balance_p4est_multirate!(mesh, dg, cache, level_info_elements, alg.stages)
# Actual move of elements across ranks
rebalance_solver!(u0, mesh, equations, dg, cache, old_global_first_quadrant)
reinitialize_boundaries!(semi.boundary_conditions, cache) # Needs to be called after `rebalance_solver!`
```
This code could then be placed in the [`resize!`](https://github.com/trixi-framework/Trixi.jl/blob/eaeb04113523500ed831e3ab459694f12f7a49ea/src/time_integration/methods_2N.jl#L251-L255) function of a corresponding multirate integrator to ensure load-balancing for simulations involving AMR.

### Boundary conditions
For p4est meshes, boundary conditions are defined and stored in dictionaries ( as shown  for example in `examples/p4est_2d_dgsem/elixir_advection_diffusion_nonperiodic_amr.jl`). If you’d like to apply the same condition to every face of the mesh, you can use the convenient functions `boundary_condition_default_p4est_2D` and `boundary_condition_default_p4est_3D`.