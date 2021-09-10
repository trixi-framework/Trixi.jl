using Trixi, OrdinaryDiffEq
using LinearAlgebra: diagm, norm
using FFTW

## Initial condition from Maulik and San 2018 (Stratified KHI paper)
# function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
#   # change discontinuity to tanh
#   # typical resolution 128^2, 256^2
#   # domain size is [-1,+1]^2
#   slope = 15
#   amplitude = 0.02
#    # discontinuous function with value 2 for -.5 <= x <= .5 and 0 elsewhere
#   B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
#   rho = 1 + 0.5 * B
#   v1 = 1.0 * (B - 1) # v1(x) ∈ [-1, 1]
#   v2 = 0.01 * sin(2 * pi * x[1])
#   p = 2.5
#   return prim2cons(SVector(rho, v1, v2, p), equations)
# end

function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 25
  amplitude = 0.02
   # discontinuous function with value 2 for -.5 <= x <= .5 and 0 elsewhere
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  # scale by (1 + .01 * sin(pi*x)*sin(pi*y)) to break symmetry
  v2 = 0.1 * sin(2 * pi * x[1]) * (1.0 + 1e-2 * sin(pi * x[1]) * sin(pi * x[2]))
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end

function compute_quantities(sol::ODESolution)
  semi = sol.prob.p
  rho_min, rho_max, p_min, p_max, S = ntuple(_->zeros(length(sol.u)), 5)
  for (i, u) in enumerate(sol.u)
    (rho_min[i], rho_max[i]), (p_min[i], p_max[i]), S[i] = compute_quantities(u, semi)
  end
  return rho_min, rho_max, p_min, p_max, S
end

function convert_Vector_to_StructArray(u, semi)
  mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
  nvars = nvariables(equations)
  uEltype = eltype(u)
  reshaped_u_raw = reinterpret(reshape, SVector{nvars, uEltype},
                               reshape(u, nvars, nnodes(dg)^2, nelements(mesh, dg, semi.cache)))
  reshaped_u = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(uEltype, size(reshaped_u_raw)), nvars))
  reshaped_u .= reshaped_u_raw

  return reshaped_u
end

function compute_quantities(u::AbstractVector, semi)
  reshaped_u = convert_Vector_to_StructArray(u, semi)
  return compute_quantities(reshaped_u, semi)
end

function integrate_entropy(u, mesh, equations, dg::DGMulti, cache)
  uq = similar(u, dg.basis.Nq, size(u, 2))
  Trixi.apply_to_each_field(Trixi.mul_by!(dg.basis.Vq), uq, u)
  S = sum(mesh.md.wJq .* entropy.(uq, equations))
  return S
end

function integrate_entropy(u, mesh, equations, dg::DGSEM, cache)
  nvars = nvariables(equations)
  reshaped_u = reinterpret(reshape, SVector{nvars, Float64}, u)
  J = 1.0 / nelements(mesh, solver, cache)
  w1D = dg.basis.weights
  wq = vec([w1D[i] * w1D[j] for j in eachnode(dg), i in eachnode(dg)])
  wJq = diagm(wq) * J * ones(nnodes(dg)^2, nelements(mesh, solver, cache))
  S = sum(wJq .* entropy.(reshaped_u, equations))
  return S
end

function compute_quantities(u::AbstractMatrix, semi)
  if typeof(u) <: StructArray
    rho = StructArrays.component(u, 1)
  else
    rho = getindex.(u, 1)
  end
  p = pressure.(u, equations)
  S = integrate_entropy(u, Trixi.mesh_equations_solver_cache(semi)...)
  return extrema(rho), extrema(p), S
end

polydeg = 3
num_cells_per_dimension = 64

# trixi_include("elixir_euler_khi_dgmulti.jl", approximation_type = SBP(),
#               tspan = (0, 3.), polydeg = polydeg, num_cells_per_dimension = num_cells_per_dimension)
trixi_include("elixir_euler_khi_DGSEM_with_Gauss_surface.jl",
              tspan = (0, 25.), polydeg = polydeg, num_cells_per_dimension = num_cells_per_dimension)
sol_DGSEM_Gauss = deepcopy(sol)
rho_min, rho_max, p_min, p_max, S = compute_quantities(sol_DGSEM_Gauss)
p1 = plot(tsave, S, label="Lobatto w/EP")
p2 = plot(tsave, 1 ./ rho_min, label="rho ratio")
plot!(p2, tsave, 1 ./ p_min, label="p ratio")

trixi_include("elixir_euler_khi_dgmulti.jl", approximation_type = GSBP(),
              tspan = (0, 25.), polydeg = polydeg, num_cells_per_dimension = num_cells_per_dimension)
sol_Gauss = deepcopy(sol)
rho_min, rho_max, p_min, p_max, S = compute_quantities(sol_Gauss)
plot!(p1, tsave, S, label = "Gauss", linestyle=:dash)
plot!(p2, tsave, 1 ./ rho_min, linestyle = :dash, label="rho ratio Gauss")
plot!(p2, tsave, 1 ./ p_min, linestyle = :dash, label="p ratio Gauss")

trixi_include("elixir_euler_khi_dgsem.jl",
              tspan = (0, 25.), polydeg = polydeg,
              initial_refinement_level = Int(log2(num_cells_per_dimension)))
sol_SBP_SC = deepcopy(sol)
rho_min, rho_max, p_min, p_max, S = compute_quantities(sol_SBP_SC)
plot!(p1, tsave, S, label="Lobatto-SC", linestyle=:dash)


# plot!(p2, tsave, 1 ./ rho_min, linestyle = :dash, label="rho ratio Gauss")
# plot!(p2, tsave, 1 ./ p_min, linestyle = :dash, label="p ratio Gauss")

plot!.((p1, p2), legend=:bottomleft)
p1

# This scales the node positions to create an equispaced grid of nodes which avoids
# boundary evaluations. The solution sampled at these points can be used to compute
# the FFT and the turbulent energy spectra
sol = sol_Gauss
sol = sol_DGSEM_Gauss

# create Vfft for DGSEM/TreeMesh
semi = sol_Gauss.prob.p
@unpack equations, mesh = semi
rd = semi.solver.basis
equispaced_fft_nodes = map(x -> x * (1 - 1 / (polydeg + 1)), StartUpDG.equi_nodes(Quad(), polydeg))
Vfft = StartUpDG.vandermonde(Quad(), polydeg, equispaced_fft_nodes...) / rd.VDM

function convert_DGMulti_to_Cartesian_product(sol)
  semi = sol.prob.p # use DGMulti semidiscretization
  u_solution = sol.u[end]

  @unpack equations, mesh = semi
  rd = semi.solver.basis
  polydeg = rd.N
  equispaced_fft_nodes = map(x -> x * (1 - 1 / (polydeg + 1)), StartUpDG.equi_nodes(Quad(), polydeg))
  Vfft = StartUpDG.vandermonde(Quad(), polydeg, equispaced_fft_nodes...) / rd.VDM

  primitive_variables = StructArrays.components(cons2prim.(u_solution, equations))
  rho, u, v, p = map(u -> Vfft * u, primitive_variables)

  # Reshapes `u` to have the data layout of a Cartesian product.
  # Assumes linear indexing along the x coordinate, then the y coordinate for both cells and nodes.
  function convert_DGMulti_to_Cartesian_product(u, polydeg, num_cells_per_dimension)
    num_pts_per_element = polydeg + 1
    Nfft = num_cells_per_dimension * num_pts_per_element
    u_cartesian = zeros(eltype(u), Nfft, Nfft)
    for ey in 1:num_cells_per_dimension
      for ex in 1:num_cells_per_dimension
        # uses explicit ordering of elements
        e = ex + (ey - 1) * num_cells_per_dimension
        u_e = reshape(view(u, :, e), polydeg + 1, polydeg + 1)'

        row_ids = (1:num_pts_per_element) .+ (ey - 1) * num_pts_per_element
        col_ids = (1:num_pts_per_element) .+ (ex - 1) * num_pts_per_element
        u_cartesian[row_ids, col_ids] .= u_e
      end
    end
    return u_cartesian
  end

  rho_cartesian, u_cartesian, v_cartesian, p_cartesian = convert_DGMulti_to_Cartesian_product.((rho, u, v, p), polydeg, num_cells_per_dimension)
  return rho_cartesian, u_cartesian, v_cartesian, p_cartesian
end

function convert_TreeMesh_to_Cartesian_product(u_raw, polydeg, num_cells_per_dimension, semi)

  mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)

  # convert to StructArray format
  x_coordinates = view(cache.elements.node_coordinates, 1, :, :, :)
  y_coordinates = view(cache.elements.node_coordinates, 2, :, :, :)
  x = reshape(x_coordinates, nnodes(dg)^2, nelements(mesh, dg, semi.cache))
  y = reshape(y_coordinates, nnodes(dg)^2, nelements(mesh, dg, semi.cache))
  u = reshape(u_raw, nnodes(dg)^2, nelements(mesh, dg, semi.cache))

  h = 1 / num_cells_per_dimension
  cell_centroids = LinRange(-1 + h, 1 - h, num_cells_per_dimension)
  mean(x) = sum(x)/length(x)

  num_pts_per_element = polydeg + 1
  Nfft = num_cells_per_dimension * num_pts_per_element
  u_cartesian = zeros(eltype(u), Nfft, Nfft)
  for e in 1:size(u, 2)

    xc = mean(view(x, :, e))
    yc = mean(view(y, :, e))
    u_e = reshape(view(u, :, e), polydeg + 1, polydeg + 1)'

    tol = 1e2*eps()
    ex = findfirst(@. abs(cell_centroids - xc) < tol)
    ey = findfirst(@. abs(cell_centroids - yc) < tol)

    row_ids = (1:num_pts_per_element) .+ (ey - 1) * num_pts_per_element
    col_ids = (1:num_pts_per_element) .+ (ex - 1) * num_pts_per_element
    u_cartesian[row_ids, col_ids] .= u_e
  end
  return u_cartesian
end

function compute_energy_spectrum(rho_cartesian, u_cartesian, v_cartesian, p)
  uhat = fft(sqrt.(rho_cartesian) .* u_cartesian)
  vhat = fft(sqrt.(rho_cartesian) .* v_cartesian)
  Ek_full = @. .5*(abs(uhat)^2 + abs(vhat)^2)

  Ek = Ek_full[1:end ÷ 2 + 1, 1:end ÷ 2 + 1] # remove duplicate modes

  wavenumbers = 1:size(Ek, 1)
  effective_wavenumbers = @. sqrt(wavenumbers^2 + wavenumbers'^2)

  N = length(wavenumbers)
  Ek_1D = zeros(N * (N - 1) ÷ 2 + N) # number of unique wavenumbers = triangular part + diagonal
  effective_wavenumber_1D = zeros(N * (N - 1) ÷ 2 + N)
  sk = 1
  for i in wavenumbers
    for j in wavenumbers[i:end] # use that the effective_wavenumber matrix is symmetric
      k = sqrt(i^2 + j^2)
      ids = findall(@. k - .5 < effective_wavenumbers < k + .5) # find wavenumbers in LinRange
      effective_wavenumber_1D[sk] = k
      Ek_1D[sk] = sum(Ek[ids])
      sk += 1
    end
  end
  return Ek_1D, effective_wavenumber_1D
end

Ek_Gauss, k1D = compute_energy_spectrum(convert_DGMulti_to_Cartesian_product(sol_Gauss)...)
Ek_DGSEM_Gauss, k1D = compute_energy_spectrum(convert_DGMulti_to_Cartesian_product(sol_DGSEM_Gauss)...)
# Ek_DGSEM_Gauss_p4, k1D_p4 = compute_energy_spectrum(convert_DGMulti_to_Cartesian_product(sol_DGSEM_Gauss_p4)...)

u_solution = convert_Vector_to_StructArray(sol_SBP_SC.u[end], sol_SBP_SC.prob.p)
primitive_variables = StructArrays.components(cons2prim.(u_solution, equations))
rho, u, v, p = map(u -> Vfft * u, primitive_variables)
rho_cartesian = convert_TreeMesh_to_Cartesian_product(rho, polydeg, num_cells_per_dimension, sol_SBP_SC.prob.p)
u_cartesian   = convert_TreeMesh_to_Cartesian_product(u, polydeg, num_cells_per_dimension, sol_SBP_SC.prob.p)
v_cartesian   = convert_TreeMesh_to_Cartesian_product(v, polydeg, num_cells_per_dimension, sol_SBP_SC.prob.p)
p_cartesian   = convert_TreeMesh_to_Cartesian_product(p, polydeg, num_cells_per_dimension, sol_SBP_SC.prob.p)
Ek_SBP, k1D = compute_energy_spectrum(rho_cartesian, u_cartesian, v_cartesian, nothing)

plot_kwargs = (; xaxis=:log, yaxis=:log, ms = 2, msw = 0)
scatter(k1D, Ek_Gauss, label="Gauss"; plot_kwargs...)
scatter!(k1D, Ek_DGSEM_Gauss, label="DGSEM with Gauss surface"; plot_kwargs...)
# scatter!(k1D_p4, Ek_DGSEM_Gauss_p4, label="DGSEM with Gauss surface (p=4)"; plot_kwargs...)
scatter!(k1D, Ek_SBP, label="DGSEM-SC-PP"; plot_kwargs...)

p = sortperm(k1D)
plot!(k1D[p], 1.5e9*k1D[p] .^ (-3), label="k^{-3}"; linestyle=:dash, legend=:bottomleft,
                                    title="Energy spectra (KHI, T = 25)")