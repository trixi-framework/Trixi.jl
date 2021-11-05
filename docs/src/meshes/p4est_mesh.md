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

#### List of corner nodes

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

#### List of elements

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

#### HOHQMesh boundary information

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

### Mesh in three spatial dimensions

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