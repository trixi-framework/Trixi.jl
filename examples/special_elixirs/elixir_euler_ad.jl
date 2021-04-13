
# This example is described in more detail in the documentation of Trixi

using Trixi, LinearAlgebra, ForwardDiff

equations = CompressibleEulerEquations2D(1.4)

mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5)

solver = DGSEM(polydeg=3, flux_lax_friedrichs, VolumeIntegralFluxDifferencing(flux_ranocha))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_isentropic_vortex, solver)

u0_ode = compute_coefficients(0.0, semi)

J = ForwardDiff.jacobian((du_ode, γ) -> begin
      equations_inner = CompressibleEulerEquations2D(first(γ))
      semi_inner = Trixi.remake(semi, equations=equations_inner, uEltype=eltype(γ));
      Trixi.rhs!(du_ode, u0_ode, semi_inner, 0.0)
    end, similar(u0_ode), [1.4]); # γ needs to be an `AbstractArray`
