"""
    init_t8code()

Initialize `t8code` by calling `sc_init`, `p4est_init`, and `t8_init` while
setting the log level to `SC_LP_ERROR`. This function will check if `t8code`
is already initialized and if yes, do nothing, thus it is safe to call it
multiple times.
"""
function init_t8code()
    # Only initialize t8code if T8code.jl can be used
    if T8code.preferences_set_correctly()
        t8code_package_id = t8_get_package_id()
        if t8code_package_id >= 0
            return nothing
        end

        # Initialize the sc library, has to happen before we initialize t8code.
        let catch_signals = 0, print_backtrace = 0, log_handler = C_NULL
            T8code.Libt8.sc_init(mpi_comm(), catch_signals, print_backtrace, log_handler,
                                 T8code.Libt8.SC_LP_ERROR)
        end

        if T8code.Libt8.p4est_is_initialized() == 0
            # Initialize `p4est` with log level ERROR to prevent a lot of output in AMR simulations
            T8code.Libt8.p4est_init(C_NULL, T8code.Libt8.SC_LP_ERROR)
        end

        # Initialize t8code with log level ERROR to prevent a lot of output in AMR simulations.
        t8_init(T8code.Libt8.SC_LP_ERROR)

        if haskey(ENV, "TRIXI_T8CODE_SC_FINALIZE")
            # Normally, `sc_finalize` should always be called during shutdown of an
            # application. It checks whether there is still un-freed memory by t8code
            # and/or T8code.jl and throws an exception if this is the case. For
            # production runs this is not mandatory, but is helpful during
            # development. Hence, this option is only activated when environment
            # variable TRIXI_T8CODE_SC_FINALIZE exists.
            @info "T8code.jl: `sc_finalize` will be called during shutdown of Trixi.jl."
            MPI.add_finalize_hook!(T8code.Libt8.sc_finalize)
        end
    else
        @warn "Preferences for T8code.jl are not set correctly. Until fixed, using `T8codeMesh` will result in a crash. " *
              "See also https://trixi-framework.github.io/Trixi.jl/stable/parallelization/#parallel_system_MPI"
    end

    return nothing
end

function trixi_t8_get_local_element_levels(forest)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(forest) != 0

    levels = Vector{Int}(undef, t8_forest_get_local_num_elements(forest))

    # Get the number of trees that have elements of this process.
    num_local_trees = t8_forest_get_num_local_trees(forest)

    current_index = 0

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        # Get the number of elements of this tree.
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(forest, itree, ielement)
            current_index += 1
            levels[current_index] = t8_element_level(eclass_scheme, element)
        end # for
    end # for

    return levels
end

# Callback function prototype to decide for refining and coarsening.
# If `is_family` equals 1, the first `num_elements` in elements
# form a family and we decide whether this family should be coarsened
# or only the first element should be refined.
# Otherwise `is_family` must equal zero and we consider the first entry
# of the element array for refinement. 
# Entries of the element array beyond the first `num_elements` are undefined.
# \param [in] forest       the forest to which the new elements belong
# \param [in] forest_from  the forest that is adapted.
# \param [in] which_tree   the local tree containing `elements`
# \param [in] lelement_id  the local element id in `forest_old` in the tree of the current element
# \param [in] ts           the eclass scheme of the tree
# \param [in] is_family    if 1, the first `num_elements` entries in `elements` form a family. If 0, they do not.
# \param [in] num_elements the number of entries in `elements` that are defined
# \param [in] elements     Pointers to a family or, if `is_family` is zero,
#                          pointer to one element.
# \return greater zero if the first entry in `elements` should be refined,
#         smaller zero if the family `elements` shall be coarsened,
#         zero else.
function adapt_callback(forest,
                        forest_from,
                        which_tree,
                        lelement_id,
                        ts,
                        is_family,
                        num_elements,
                        elements)::Cint
    num_levels = t8_forest_get_local_num_elements(forest_from)

    indicator_ptr = Ptr{Int}(t8_forest_get_user_data(forest))
    indicators = unsafe_wrap(Array, indicator_ptr, num_levels)

    offset = t8_forest_get_tree_element_offset(forest_from, which_tree)

    # Only allow coarsening for complete families.
    if indicators[offset + lelement_id + 1] < 0 && is_family == 0
        return Cint(0)
    end

    return Cint(indicators[offset + lelement_id + 1])
end

function trixi_t8_adapt_new(old_forest, indicators)
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    let set_from = C_NULL, recursive = 0, no_repartition = 1, do_ghost = 1
        t8_forest_set_user_data(new_forest, pointer(indicators))
        t8_forest_set_adapt(new_forest, old_forest, @t8_adapt_callback(adapt_callback),
                            recursive)
        t8_forest_set_balance(new_forest, set_from, no_repartition)
        t8_forest_set_ghost(new_forest, do_ghost, T8_GHOST_FACES)
        t8_forest_commit(new_forest)
    end

    return new_forest
end

function trixi_t8_get_difference(old_levels, new_levels, num_children)
    old_nelems = length(old_levels)
    new_nelems = length(new_levels)

    changes = Vector{Int}(undef, old_nelems)

    # Local element indices.
    old_index = 1
    new_index = 1

    while old_index <= old_nelems && new_index <= new_nelems
        if old_levels[old_index] < new_levels[new_index]
            # Refined.

            changes[old_index] = 1

            old_index += 1
            new_index += num_children

        elseif old_levels[old_index] > new_levels[new_index]
            # Coarsend.

            for child_index in old_index:(old_index + num_children - 1)
                changes[child_index] = -1
            end

            old_index += num_children
            new_index += 1

        else
            # No changes.

            changes[old_index] = 0

            old_index += 1
            new_index += 1
        end
    end

    return changes
end

# Coarsen or refine marked cells and rebalance forest. Return a difference between
# old and new mesh.
function trixi_t8_adapt!(mesh, indicators)
    old_levels = trixi_t8_get_local_element_levels(mesh.forest)

    forest_cached = trixi_t8_adapt_new(mesh.forest, indicators)

    new_levels = trixi_t8_get_local_element_levels(forest_cached)

    differences = trixi_t8_get_difference(old_levels, new_levels, 2^ndims(mesh))

    mesh.forest = forest_cached

    return differences
end
