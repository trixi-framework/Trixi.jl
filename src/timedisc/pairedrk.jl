module PairedRk

export calc_coefficients

function calc_coefficients(n_stages, n_derivative_evaluations)
  if !haskey(rk_a, (n_stages, n_derivative_evaluations))
    error("no coefficients stored for n_stages=$n_stages, " *
          "n_derivative_evaluations=$n_derivative_evaluations")
  end

  # c is equidistant from zero to 1/2
  c = zeros(n_stages)
  for i in 1:n_stages
    c[i] = (i - 1)/(2 * (n_stages - 1))
  end

  # Calculate a
  a = zeros(n_stages, n_stages)

  # First, store get the pre-computed coefficients
  for i in (2 + n_stages - n_derivative_evaluations + 1):n_stages
    a[i, i-1] = rk_a[n_stages, n_derivative_evaluations][i, i-1]
  end

  # Then, calculate the first column of the Butcher tableau
  for i in 2:n_stages
    a[i, 1] = c[i] - a[i, i-1]
  end

  return a, c
end


# `a[s, e]` stores the Runge-Kutta coefficients `a` for a given combination of
# stages `s` and derivative evaluations `e`. Only the coefficients determined
# by the optimization routines are stored, e.g., for a (s,e) = (8,4) scheme,
# the stored coefficients are `a(6,5)`, `a(7,6)`, and `a(8,7)`.
Coeff = Dict{Tuple{Int, Int}, Float64}
rk_a = Dict{Tuple{Int, Int}, Coeff}()

# s = 8, e = 8
rk_a[ 8, 8] = Coeff()
rk_a[ 8, 8][ 8, 7] = 0.37579012910525006
rk_a[ 8, 8][ 7, 6] = 0.26593425769133050
rk_a[ 8, 8][ 6, 5] = 0.19302247744289253
rk_a[ 8, 8][ 5, 4] = 0.13904874358308181
rk_a[ 8, 8][ 4, 3] = 9.6034360454394327E-002
rk_a[ 8, 8][ 3, 2] = 5.9894669110050543E-002

# s = 16, e = 16
rk_a[16,16] = Coeff()
rk_a[16,16][16,15] = 0.35393625497817993
rk_a[16,16][15,14] = 0.26191739228182914
rk_a[16,16][14,13] = 0.20481028466693352
rk_a[16,16][13,12] = 0.16524992419475620
rk_a[16,16][12,11] = 0.13578562845844364
rk_a[16,16][11,10] = 0.11267892976337304
rk_a[16,16][10, 9] = 9.3843410496628157E-002
rk_a[16,16][ 9, 8] = 7.8018402529704295E-002
rk_a[16,16][ 8, 7] = 6.4395286545402816E-002
rk_a[16,16][ 7, 6] = 5.2430332461051193E-002
rk_a[16,16][ 6, 5] = 4.1743665531295090E-002
rk_a[16,16][ 5, 4] = 3.2061621537737894E-002
rk_a[16,16][ 4, 3] = 2.3182014680455002E-002
rk_a[16,16][ 3, 2] = 1.4952303950403279E-002

# s = 16, e = 8
rk_a[16, 8] = Coeff()
rk_a[16, 8][16,15] = 0.34511338387216839
rk_a[16, 8][15,14] = 0.23865894921016836
rk_a[16, 8][14,13] = 0.16728614711717354
rk_a[16, 8][13,12] = 0.11376715384070328
rk_a[16, 8][12,11] = 7.0425197666555847E-002
rk_a[16, 8][11,10] = 3.3274816172250303E-002

# s = 16, e = 4
rk_a[16, 4] = Coeff()
rk_a[16, 4][16,15] = 0.30860538993562969
rk_a[16, 4][15,14] = 0.13495465515328439

# s = 16, e = 2 -> no unknown coefficients necessary but included for consistency
rk_a[16, 2] = Coeff()


end # module PairedRk
