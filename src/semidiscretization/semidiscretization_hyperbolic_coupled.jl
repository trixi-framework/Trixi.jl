struct SemidiscretizationHyperbolicCoupled{S, I} <: AbstractSemidiscretization
  semis::S
  u_indices::I
end

function SemidiscretizationHyperbolicCoupled(semis)
  n_coeffs = semis .|> mesh_equations_solver_cache .|> (x -> n_coefficients(x...)) |> collect
  u_indices = Vector{UnitRange{Int}}(undef, length(semis))

  for i in 1:length(semis)
    offset = sum(n_coeffs[1:i-1]) + 1
    u_indices[i] = range(offset, length=n_coeffs[i])
  end

  SemidiscretizationHyperbolicCoupled{typeof(semis), typeof(u_indices)}(semis, u_indices)
end


@inline nmeshes(semi::SemidiscretizationHyperbolicCoupled) = length(semi.semis)
@inline Base.real(semi::SemidiscretizationHyperbolicCoupled) = real(semi.semis[1])


function compute_coefficients(t, semi::SemidiscretizationHyperbolicCoupled)
  @unpack u_indices = semi

  u_ode = Vector{real(semi)}(undef, u_indices[end][end])

  for i in 1:nmeshes(semi)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    u_ode[u_indices[i]] .= compute_coefficients(semi.semis[i].initial_condition, t, semi.semis[i])
  end
  
  return u_ode
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicCoupled, t)
  @unpack u_indices = semi

  for i in 1:nmeshes(semi)
    u_loc  = @view u_ode[u_indices[i]]
    du_loc = @view du_ode[u_indices[i]]

    rhs!(du_loc, u_loc, semi.semis[i], t)
  end

  return nothing
end
