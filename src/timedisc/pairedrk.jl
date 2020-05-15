module PairedRk

export calc_coefficients
export calc_c
export calc_a_multilevel
export acc_level_ids_by_stage


function calc_coefficients(polyDeg, n_stages, n_derivative_evaluations)
  if !haskey(rk_a, (polyDeg, n_stages, n_derivative_evaluations))
    error("no coefficients stored for polyDeg=$polyDeg, n_stages=$n_stages, " *
          "n_derivative_evaluations=$n_derivative_evaluations")
  end

  # Get c
  c = calc_c(n_stages)

  # Calculate a
  a = zeros(n_stages, n_stages)

  # First, get the pre-computed coefficients
  for i in (2 + n_stages - n_derivative_evaluations + 1):n_stages
    a[i, i-1] = rk_a[polyDeg, n_stages, n_derivative_evaluations][i, i-1]
  end

  # Then, calculate the first column of the Butcher tableau
  for i in 2:n_stages
    a[i, 1] = c[i] - a[i, i-1]
  end

  return a, c
end


function calc_c(n_stages::Integer)
  # c is equidistant from zero to 1/2
  c = zeros(n_stages)
  for i in 1:n_stages
    c[i] = (i - 1)/(2 * (n_stages - 1))
  end

  return c
end


# Calculate Runge-Kutta coefficients for multilevel P-ERK scheme
#
# Note: If at the coarsest level more stage evaluations are to be activated,
#       you need to modify the two locations marked (**)
function calc_a_multilevel(polyDeg, n_stages, stage, n_derivative_evaluations_max,
                           n_elements, level_info_elements)
  # Determine number of levels
  n_levels = length(level_info_elements)

  # Use at least two derivative evaluations
  n_derivative_evaluations_min = 2

  # Sanity check
  @assert n_derivative_evaluations_max in (2, 4, 8, 16) "maximum number of derivative evaluations" *
      "is not in (2, 4, 8, 16)"

  # Store number of derivative evaluations for each level id
  e_by_level_id = Vector{Int}(undef, n_levels)
  e = n_derivative_evaluations_max
  for level_id in 1:n_levels
    e_by_level_id[level_id] = e
    if e == 16
      e = 8
    elseif e == 8
      e = 4
    elseif e == 4
      e = 2 # (**) Set this to, e.g., 3 to have at least three evaluations
    end
  end

  # Allocate storage for coefficients (a_1 = aₛ₁, a_2 = aₛ,ₛ₋₁)
  a_1 = fill(NaN, n_elements)
  a_2 = fill(NaN, n_elements)

  # Determine coefficients for each element by level
  for level_id in 1:n_levels
    a, _ = calc_coefficients(polyDeg, n_stages, e_by_level_id[level_id])
    a_1[level_info_elements[level_id]] .= a[stage, 1]
    a_2[level_info_elements[level_id]] .= a[stage, stage-1]
  end

  return a_1, a_2
end


function acc_level_ids_by_stage(n_stages, n_levels)
  # Init storage
  acc_level_ids = Vector{Int}(undef, n_stages)

  # Fill storage depending on number of stages
  if n_stages == 16
    acc_level_ids[ 1] = n_levels
    acc_level_ids[ 2] = 1
    acc_level_ids[ 3] = 1
    acc_level_ids[ 4] = 1
    acc_level_ids[ 5] = 1
    acc_level_ids[ 6] = 1
    acc_level_ids[ 7] = 1
    acc_level_ids[ 8] = 1
    acc_level_ids[ 9] = 1
    acc_level_ids[10] = 2
    acc_level_ids[11] = 2
    acc_level_ids[12] = 2
    acc_level_ids[13] = 2
    acc_level_ids[14] = 3
    acc_level_ids[15] = 3 # (**) Set to `n_levels` to have three evaluations at coarsest level
    acc_level_ids[16] = n_levels
  else
    error("number of stages '$n_stages' not supported")
  end

  # Return minimum of calculated level and number of levels to account for
  # cases where more levels are supported than currently exist
  return min.(acc_level_ids, n_levels)
end


# `rk_a[N, s, e]` stores the Runge-Kutta coefficients `a` for a given combination of
# polynomial degree `N`, stages `s`, and derivative evaluations `e`. Only the
# coefficients determined by the optimization routines are stored, e.g., for a
# (N,s,e) = (4,8,4) scheme, the stored coefficients are `a(6,5)`, `a(7,6)`, and
# `a(8,7)`.
Coeff = Dict{Tuple{Int, Int}, Float64}
rk_a = Dict{Tuple{Int, Int, Int}, Coeff}()

################################################################################
# Polynomial degree N = 4
################################################################################
# s = 8, e = 8
rk_a[4, 8, 8] = Coeff()
rk_a[4, 8, 8][ 8, 7] = 0.37579012910525006
rk_a[4, 8, 8][ 7, 6] = 0.26593425769133050
rk_a[4, 8, 8][ 6, 5] = 0.19302247744289253
rk_a[4, 8, 8][ 5, 4] = 0.13904874358308181
rk_a[4, 8, 8][ 4, 3] = 9.6034360454394327E-002
rk_a[4, 8, 8][ 3, 2] = 5.9894669110050543E-002

# s = 16, e = 16
rk_a[4, 16,16] = Coeff()
rk_a[4, 16,16][16,15] = 0.35393625497817993
rk_a[4, 16,16][15,14] = 0.26191739228182914
rk_a[4, 16,16][14,13] = 0.20481028466693352
rk_a[4, 16,16][13,12] = 0.16524992419475620
rk_a[4, 16,16][12,11] = 0.13578562845844364
rk_a[4, 16,16][11,10] = 0.11267892976337304
rk_a[4, 16,16][10, 9] = 9.3843410496628157E-002
rk_a[4, 16,16][ 9, 8] = 7.8018402529704295E-002
rk_a[4, 16,16][ 8, 7] = 6.4395286545402816E-002
rk_a[4, 16,16][ 7, 6] = 5.2430332461051193E-002
rk_a[4, 16,16][ 6, 5] = 4.1743665531295090E-002
rk_a[4, 16,16][ 5, 4] = 3.2061621537737894E-002
rk_a[4, 16,16][ 4, 3] = 2.3182014680455002E-002
rk_a[4, 16,16][ 3, 2] = 1.4952303950403279E-002

# s = 16, e = 8
rk_a[4, 16, 8] = Coeff()
rk_a[4, 16, 8][16,15] = 0.34511338387216839
rk_a[4, 16, 8][15,14] = 0.23865894921016836
rk_a[4, 16, 8][14,13] = 0.16728614711717354
rk_a[4, 16, 8][13,12] = 0.11376715384070328
rk_a[4, 16, 8][12,11] = 7.0425197666555847E-002
rk_a[4, 16, 8][11,10] = 3.3274816172250303E-002

# s = 16, e = 4
rk_a[4, 16, 4] = Coeff()
rk_a[4, 16, 4][16,15] = 0.30860538993562969
rk_a[4, 16, 4][15,14] = 0.13495465515328439

# s = 16, e = 3
rk_a[4, 16, 3] = Coeff()
rk_a[4, 16, 3][16,15] = 0.27284927666187286

# s = 16, e = 2 -> no unknown coefficients necessary but included for consistency
rk_a[4, 16, 2] = Coeff()


end # module PairedRk
