# Trixi Development Notes

## Ready for rock'n'roll

1.  Separate `differentiation_matrix` into individual matrices and keep all
    matrices around all the time (i.e., `D`, `Dhat`, `Dsplit`)
2.  Rename `u_t` to `ut`
3.  `cons2prim`, `cons2entropy` etc. should be implemented point-wise in
    equations, element-wise in solver
4.  Rename stuff
    *   `weights` -> `weights1d` (also add `weights2d`)
    *   `nodes` -> `nodes1d` (also add `nodes2d`)
    *   `inverse_jacobian` -> `inverse_jacobian1d` (also add `jacobian1d`,
        `jacobian2d`
    *   All "getter" methods that obtain a certain attribute from an object
        should be prefixed by `get` such that the original name is free to be used
        as the variable name.  
        Examples: `nnodes`, `polydeg`, `nelements`, `nsurfaces`,
5.  Get rid of `n_surfaces`/`n_elements` etc. variable for which a `get_xxx`
    function exists.
6.  Fix `--interactive` mode for Trixi.


## Open issues with unresolved questions or unknown implementation

1.  How to avoid clutter with many arguments being passed to methods (i.e.,
    passing just `dg` vs `dg.u`, `dg.ut`, ...)?
2.  Optimize file/directory structure, e.g.,
    *   split equations into multiple files/folders?
    *   split DG methods in multiple files?
3.  **Performance optimization**
    a.  Reduce memory allocations in hot kernels
    b.  Improve performance by using @inbounds (?)
4.  Implement logging mechanism: basically store all output in log file such
    that it can be recovered later, e.g., in `out/trixi_YYYMMDDHHmmss.log`
5.  We need a testing tool for easy testing of functionality in case of larger
    refactorings and/or new features that affect existing solutions
6.  **Visualization**
    a.  Visualize element data (e.g., one data point per element)
