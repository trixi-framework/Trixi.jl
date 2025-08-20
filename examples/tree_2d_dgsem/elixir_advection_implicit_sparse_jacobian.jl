using Trixi
using OrdinaryDiffEqSDIRK

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

###############################################################################################
### Overloads to construct the `LobattoLegendreBasis` with `Real` type (supertype of `Num`) ###

# Required for setting up the Lobatto-Legendre basis for abstract `Real` type.
# Constructing the Lobatto-Legendre basis with `Real` instead of `Num` is 
# significantly easier as we do not have to care about e.g. if-clauses.
# As a consequence, we need to provide some overloads hinting towards the intended behavior.

const float_type = Float64 # Actual floating point type for the simulation

# Newton tolerance for finding LGL nodes & weights
Trixi.eps(::Type{Real}) = Base.eps(float_type)
# There are some places where `one(RealT)` or `zero(uEltype)` is called where `RealT` or `uEltype` is `Real`.
# This returns an `Int64`, i.e., `1` or `0`, respectively which gives errors when a floating-point alike type is expected.
Trixi.one(::Type{Real}) = Base.one(float_type)
Trixi.zero(::Type{Real}) = Base.zero(float_type)

module RealMatMulOverload

# Multiplying two Matrix{Real}s gives a Matrix{Any}.
# This causes problems when instantiating the Legendre basis, which calls
# `calc_{forward,reverse}_{upper, lower}` which in turn uses the matrix multiplication
# which is overloaded here in construction of the interpolation/projection operators 
# required for mortars.
function Base.:*(A::Matrix{Real}, B::Matrix{Real})::Matrix{Real}
    m, n = size(A, 1), size(B, 2)
    kA = size(A, 2)
    kB = size(B, 1)
    @assert kA==kB "Matrix dimensions must match for multiplication"

    C = Matrix{Real}(undef, m, n)
    for i in 1:m, j in 1:n
        #acc::Real = zero(promote_type(typeof(A[i,1]), typeof(B[1,j])))
        acc = zero(Real)
        for k in 1:kA
            acc += A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    return C
end
end

import .RealMatMulOverload

###############################################################################################
### semidiscretizations of the linear advection equation ###

advection_velocity = (0.2, -0.7)
equation = LinearScalarAdvectionEquation2D(advection_velocity)

# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num` from Symbolics
# `solver_real` is used for computing the Jacobian sparsity pattern
solver_real = DGSEM(polydeg = 3, surface_flux = flux_godunov, RealT = Real)
# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 3, surface_flux = flux_godunov)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

# `semi_real` is used for computing the Jacobian sparsity pattern
semi_real = SemidiscretizationHyperbolic(mesh, equation,
                                         initial_condition_convergence_test,
                                         solver_real)
# `semi_float` is  used for the subsequent simulation
semi_float = SemidiscretizationHyperbolic(mesh, equation,
                                          initial_condition_convergence_test,
                                          solver_float)

t0 = 0.0 # Re-used for the ODE function defined below
t_end = 1.0
t_span = (t0, t_end)

# Call `semidiscretize` on `semi_float` to create the ODE problem to have access to the initial condition.
# For the linear example considered here one could also use an arbitrary vector for the initial condition.
ode_float = semidiscretize(semi_float, t_span)
u0_ode = ode_float.u0
du_ode = similar(u0_ode)

###############################################################################################
### Compute the Jacobian with SparseDiffTools ###

# Create a function with two parameters: `du_ode` and `u0_ode`
# to fulfill the requirements of an in_place function in SparseDiffTools
# (see example function `f` from https://docs.sciml.ai/SparseDiffTools/dev/#Example)
rhs = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_real, t0)

# Taken from example linked above to detect the pattern and choose how to do the AutoDiff automatically
sd = SymbolicsSparsityDetection()
ad_type = AutoFiniteDiff() # `AutoForwardDiff()` does also work, but not strictly necessary for sparsity detection only
sparse_adtype = AutoSparse(ad_type)

# `sparse_cache` will reduce calculation time when Jacobian is calculated multiple times,
# which is in principle not required for the linear problem considered here.
sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs, du_ode, u0_ode)

###############################################################################################
### Set up sparse-aware ODEProblem ###

# Revert overrides from above for the actual simulation -
# not strictly necessary, but good practice
Trixi.eps(x::Type{Real}) = Base.eps(x)
Trixi.one(x::Type{Real}) = Base.one(x)
Trixi.zero(x::Type{Real}) = Base.zero(x)

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_float_jac_sparse = semidiscretize(semi_float, t_span,
                                      jac_prototype = sparse_cache.jac_prototype,
                                      colorvec = sparse_cache.coloring.colorvec)

# Note: We tried for this linear problem providing the constant, sparse Jacobian directly via
#
# const jac_sparse = sparse_jacobian(sparse_adtype, sparse_cache, rhs, du_ode, u0_ode)
# function jac_sparse_func!(J, u, p, t)
#   J = jac_sparse
# end
# SciMLBase.ODEFunction(rhs!, jac_prototype=float.(jac_prototype), colorvec=colorvec, jac = jac_sparse_func!)
#
# which turned out (for the considered problems) to be significantly slower than 
# just using the prototype and the coloring vector.

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi_float, interval = 10)
save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

# Note: No `stepsize_callback` due to (implicit) solver with adaptive timestep control
callbacks = CallbackSet(summary_callback, analysis_callback, save_restart)

###############################################################################################
### solve the ODE problem ###

sol = solve(ode_float_jac_sparse, # using `ode_float_jac_sparse` instead of `ode_float` results in speedup of factors 10-15!
            # Default `AutoForwardDiff()` is not yet working,
            # probably related to https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#Autodifferentiation-and-Dual-Numbers
            TRBDF2(; autodiff = ad_type);
            dt = 0.1, save_everystep = false, callback = callbacks);
