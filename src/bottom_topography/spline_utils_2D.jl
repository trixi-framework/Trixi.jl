# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

##################################################
### Help functions for 2D spline interpolation ###
##################################################

# Sort data so that x and y values are ascending and that the matrix z contains
# the corresponding values.
"""
    sort_data(x::Vector, y::Vector, z::Matrix)

This function sorts the inputs vectors `x` and `y` in a ascending order
and also reorders the input matrix `z` accordingly. 

Therefore first the `x` values are sorted with the matrix `z` accordingly and
afterwards the `y` values are sorted with the matrix `z` accordingly.

The sorted `x`, `y` and `z` values are returned.
"""
function sort_data(x::Vector, y::Vector, z::Matrix)

  zx              = transpose(z)
  original_data_x = hcat(x, zx)
  sorted_data_x   = original_data_x[sortperm(original_data_x[:,1]), :]

  x_sorted  = sorted_data_x[:,1]
  z_interim = sorted_data_x[:,2:end]

  zy              = transpose(z_interim)
  original_data_y = hcat(y, zy)
  sorted_data_y   = original_data_y[sortperm(original_data_y[:,1]), :]

  y_sorted = sorted_data_y[:,1]
  z_sorted = sorted_data_y[:,2:end]
  
  return x_sorted, y_sorted, Matrix(z_sorted)
end

###################################
### Thin plate spline smoothing ###
###################################

# Base function for thin plate spline
"""
    tps_base_func(r::Number)

Thin plate spline basis function.

- Gianluca Donato and Serge Belongie (2001)
  Approximate Thin Plate Spline Mappings
  [DOI: 10.1007/3-540-47977-5_2](https://link.springer.com/content/pdf/10.1007/3-540-47977-5_2.pdf)
"""
function tps_base_func(r::Number)

  if r == 0
    return 0
  else
    return r*r*log(r)
  end
    
end

# restructure input data to be able to use the thin plate spline functionality
@doc raw"""
      restructure_data(x::Vector, y::Vector, z::Matrix)

This function restructures the input values
- `x`: a vector with `n` values in x-direction
- `y`: a vector with `m` values in y-direction
- `z`: a  `m` ``\\times`` `n` matrix with values in z-direction where the values of `z` correspond 
       to the indexing `(y,x)`

The output is of the following form:
```math
\begin{equation}
  \begin{bmatrix}
    x_1 & y_1 & z_{1,1}\\
    x_2 & y_1 & z_{1,2}\\
    & \vdots & \\
    x_n & y_1 & z_{1,n}\\
    x_1 & y_2 & z_{2,1}\\
    & \vdots & \\
    x_n & y_m & z_{m,n}
  \end{bmatrix}
\end{equation}
```

"""
function restructure_data(x::Vector, y::Vector, z::Matrix)

  x_mat = repeat(x, 1, length(y))
  y_mat = repeat(y, 1, length(x))
  
  p = length(z)

  x_vec = vec(reshape(x_mat , (p, 1)))
  y_vec = vec(reshape(y_mat', (p, 1)))
  z_vec = vec(reshape(z'    , (p, 1)))

  return [x_vec y_vec z_vec]
end

# Thin plate spline approximation  
@doc raw"""
    calc_tps(lambda::Number, x::Vector, y::Vector, z::Matrix)

The inputs to this function are:
- `lambda`: a smoothing factor which specifies the degree of the smoothing that should take place
- `x`: a vector of `x` values 
- `y`: a vector of `x` values
- `z`: a matrix with the `z` values to be smoothed where the values of `z` correspond to the
       indexing `(y,x)`

This function uses the thin plate spline approach to perform the smoothing.
To do so the following linear equations system has to be solved for `coeff`:
```math
\begin{equation}\label{tps_mat}
		\underbrace{
		\begin{bmatrix}
			K & P \\
			P^T & O
		\end{bmatrix}
		}_{:= L}
		\underbrace{\begin{bmatrix}
			w \\ a
		\end{bmatrix}}_{\text{:= coeff}}
		=
		\underbrace{\begin{bmatrix}
			z\\o
		\end{bmatrix}}_{\text{:= rhs}}
\end{equation}
```
First of all the inputs are restructured using the function [`restructure_data`](@ref) and
saved in the variables `x_hat`, `y_hat` and `z_hat`.

Then the matrix `L` can be filled by setting 
`K` = [`tps_base_func`](@ref)`(||(x_hat[i], y_hat[i]) - (x_hat[j], y_hat[j])||)` where `|| ||` is 
the Eucledian norm, `P` = `[1 x y]` and `O` = ``3\times 3`` zeros matrix.

Afterwards the vector `rhs` is filled by setting `z` = `z_hat` and `o` = a vector with three zeros.

Now the system is solved to redeem the vector `coeff`.
This vector is then used to calculate the smoothed values for `z` and save them in `H_f` by 
the following function:
```math
\begin{align}
H\_f[i] = &a[1] + a[2]x\_hat[i] + a[3]y\_hat[i] \\
        + &\sum_{j = 0}^p tps\_base\_func(\|(x\_hat[i], y\_hat[i]) - (x\_hat[j], y\_hat[j]) \|)
\end{align}
```
here `p` is the number of entries in `z_hat`.

A reference to the calculations can be found in the lecture notes of
- Gianluca Donato and Serge Belongie (2001)
  Approximate Thin Plate Spline Mappings
  [DOI: 10.1007/3-540-47977-5_2](https://link.springer.com/content/pdf/10.1007/3-540-47977-5_2.pdf)
"""
function calc_tps(lambda::Number, x::Vector, y::Vector, z::Matrix)

  restructured_data = restructure_data(x,y,z)
  x_hat = restructured_data[:,1]
  y_hat = restructured_data[:,2]
  z_hat = restructured_data[:,3]

  n = length(x)
  m = length(y)
  p = length(z)

  L   = zeros(p+3, p+3)
  rhs = zeros(p+3, 1  )
  H_f = zeros(p  , 1  )

  # Fill K part of matrix L
  for i in 1:p
    for j in (i+1):p
      p_i    = [x_hat[i], y_hat[i]]
      p_j    = [x_hat[j], y_hat[j]]
      U      = tps_base_func(norm(p_i .- p_j))
      L[i,j] = U
      L[j,i] = U
    end
  end

  # Fill rest of matrix L
  L[1:p,1:p] = L[1:p,1:p] + lambda * diagm(ones(p))
  
  L[1:p,p+1] = ones(p)
  L[1:p,p+2] = x_hat
  L[1:p,p+3] = y_hat

  L[p+1,1:p] = ones(p)
  L[p+2,1:p] = x_hat
  L[p+3,1:p] = y_hat

  # Fill part z of rhs
  rhs[1:p,1] = z_hat

  # Calculate solution vector
  coeff = L\rhs

  # Fill matrix grid with smoothed z values
  for i in 1:p
    H_f[i] = coeff[p+1] + coeff[p+2]*x_hat[i] + coeff[p+3]*y_hat[i]
    p_i    = [x_hat[i], y_hat[i]]
    for k in 1:p
      p_k    = [x_hat[k], y_hat[k]]
      H_f[i] = H_f[i] + coeff[k] * tps_base_func(norm(p_i .- p_k))
    end
  end

  return transpose(reshape(H_f, (n,m)))
end

end # @muladd