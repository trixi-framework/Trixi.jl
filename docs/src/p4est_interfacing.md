# Interfacing with the `p4est` mesh backend

Sometimes it is necessary to exchange data between the solver and the mesh.
In the case of the [`P4estMesh`](@ref) and its [`p4est`](https://www.p4est.org/), this is more intricate than with other mesh types, since the `p4est` backend
is comparatively opaque to the rest of the code due to being a compiled C library.

As a disclaimer: This is neither complete nor self-contained, and is meant as a first orientation guide.
To implement own features, you likely need to take a look at the documentation of [`p4est`](https://github.com/cburstedde/p4est), its Julia wrapper [P4est.jl](https://github.com/trixi-framework/P4est.jl), and relevant parts of the [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) source code.
Additionally, the [Julia manual for interfacing C code](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/) and the [documentation of the Julia C interface](https://docs.julialang.org/en/v1/base/c/) are also helpful.

First, we go through the implementation of adaptive mesh refinement (AMR) with `p4est`.
Second, we show how a custom mesh partitioning based on a weighting function can be implemented.

## Adaptive Mesh Refinement (AMR)

In a simulation that uses adaptive mesh refinement (AMR), cells are assigned an indicator value $I \in \{-1, 0, 1\}$ that denotes whether the cell should be coarsened ($I = -1$), kept at the current level ($I = 0$), or be refined ($I = 1$).

For realizing AMR there are now in principle two possibilities:
- First, the solver could receive the connectivity information from the mesh backend and then handle the refinement and coarsening of the mesh.
- Second, the solver could send the indicator values to the mesh backend and let it handle the refinement and coarsening of the mesh.

For the `p4est` mesh, Trixi.jl uses the latter approach to benefit from the highly efficient implementation of the `p4est` library.

The first step involves sending the indicator values to the mesh backend.
The indicator values `lambda` are obtained from a "controller" in the `AMRCallback`
```julia
lambda = @trixi_timeit timer() "indicator" controller(u, mesh, equations, dg, cache,
                                                      t = t, iter = iter)
```
and need now to be sent to the mesh backend.
To this end, a number of helper functions are required.
First, a Julia function with two arguments `info, user_data` is defined that copies the indicator values from `user_data` to a field in `info`, which is a field from the `p4est` mesh (precisely `p4est_iter_volume_info_t` - for details see the [`p4est` documentation](https://p4est.github.io/api/p4est-latest/p4est__iterate_8h.html)):
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
This function is then converted to a C function to match the required syntax of the [`p4est_iter_volume_t`](https://p4est.github.io/api/p4est-latest/p4est__iterate_8h.html)/[`p8est_iter_volume_t`](https://p4est.github.io/api/p4est-latest/p8est__iterate_8h.html#a4b19423fb264c674bd4deaf5a7194758) function:
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
To handle both 2D (which is the canonical `p4est`) and 3D (which is within the `p4est` library referred to as `p8est`), we also dispatch on the number of spatial dimensions:
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

The correct function is then selected via
```julia
iter_volume_c = cfunction(copy_to_quad_iter_volume, Val(ndims(mesh)))
```
and by calling `iterate_p4est` the indicator values are finally copied to the cells (quadrants or octants) of the mesh:
```julia
iterate_p4est(mesh.p4est, # The p4est mesh
              lambda; # The indicator values
              # The function that copies the indicator values to the mesh
              iter_volume_c = iter_volume_c)
```

Now, during the `refine!` or `coarsen!` calls (which eventually call [`refine_fn`](https://p4est.github.io/api/p4est-latest/p4est_8h.html#a1a31375edfa42b5609e7656233c32cca)/[`coarsen_fn`](https://p4est.github.io/api/p4est-latest/p4est_8h.html#ad250f4765d9778ec3940e9fabea7c853) from `p4est`) the indicator values are available in the user data of the mesh cells.
In particular, again a function needs to be provided that matches the signature of the [`p4est_refine_t`](https://p4est.github.io/api/p4est-latest/p4est_8h.html#ad6f6d433abde78f20ea267e6aebea26a)/[`p8est_refine_t`](https://p4est.github.io/api/p4est-latest/p8est_8h.html#a24565b65860e156a04ba8ccc6f67a936) function
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
which is implemented as 
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
Again, dimensional dispatching is used to handle both 2D and 3D meshes:
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
These are then handed over to `refine_p4est!` which eventually calls `refine_fn`.

## Custom mesh partitioning

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

Similar to the `refine_fn` function, `p4est` requires a [weighting function with the same signature](https://p4est.github.io/api/p4est-latest/p4est_8h.html#aa03358f1326e23d122ef1b155705fd4d):
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
Again, the [3D version](https://p4est.github.io/api/p4est-latest/p8est_8h.html#a065466172704df28878d8535b98965a1) is very similar:
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
Thus, the call to `partition!` needs to be followed by a call to `rebalance_solver!` to rebalance the solver data.
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