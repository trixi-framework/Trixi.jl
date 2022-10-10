# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

##############################################
### Two dimensional B-spline interpolation ###
##############################################

# Bilinaer B-spline interpolation
@doc raw"""
    spline_interpolation(b_spline::BilinearBSpline, u, v)

The inputs are the [`BilinearBSpline`](@ref) object and the variable `u` and `v` at which the spline 
will be evaluated. Where `u` corresponds to a value in x-direction and `v` to a value in 
y-direction.

The parameters `i` and `j` give us the patch in which `(u,v)` is located.
Which will also be used to get the correct control points from `Q`.
(A patch is the area between two consecutive `x` and `y` values)

`my` is  an interim variable which maps `u` to the interval ``[0,1]``
for further calculations. `ny` does the same for `v`.

To evaluate the spline at `(u,v)`, we have to calculate the following:
```math
\begin{equation}
c_{i,j,1}(\mu_i(u),\nu_j(v)) = 
		\begin{bmatrix} \mu_i(u)\\ 1 \end{bmatrix}^T
		\underbrace{\begin{bmatrix} -1 & 1\\ 1 & 0 \end{bmatrix}}_{\text{IP}}
		\begin{bmatrix} Q_{i,j} & Q_{i,j+1}\\ Q_{i+1,j} & Q_{i+1,j+1} \end{bmatrix}
		\underbrace{\begin{bmatrix} -1 & 1\\ 1 & 0 \end{bmatrix}}_{\text{IP}^T}
		\begin{bmatrix} \nu_j(v) \\ 1\end{bmatrix}
\end{equation}
```

A reference for the calculations in this script can be found in Chapter 2 of
-  Quentin Agrapart & Alain Batailly (2020)
   Cubic and bicubic spline interpolation in Python. 
   [hal-03017566v2](https://hal.archives-ouvertes.fr/hal-03017566v2)
"""
function spline_interpolation(b_spline::BilinearBSpline, u, v)

  x  = b_spline.x
  y  = b_spline.y
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP

  i = max(1, min(searchsortedlast(x, u), length(x) - 1))
  j = max(1, min(searchsortedlast(y, v), length(y) - 1))

  my = (u - x[i])/h
  ny = (v - y[j])/h

  Q_temp = [Q[i, j:(j+1)] Q[(i+1), j:(j+1)]]

  c = [ny, 1]' * IP * Q_temp * IP' * [my, 1]
  
  return c
end

# Bicubic B-spline interpolation
@doc raw"""
    spline_interpolation(b_spline::BicubicBSpline, u, v)

The inputs are the [`BicubicBSpline`](@ref) object and the variable `u` and `v` at which the spline 
will be evaluated. Where `u` corresponds to a value in x-direction and `v` to a value in 
y-direction.

The parameters `i` and `j` give us the patch in which `(u,v)` is located.
Which will also be used to get the correct control points from `Q`.
(A patch is the area between two consecutive `x` and `y` values)

`my` is  an interim variable which maps `u` to the interval ``[0,1]``
for further calculations. `ny` does the same for `v`.

To evaluate the spline at `(u,v)`, we have to calculate the following:
```math
\begin{equation}
c_{i,j,3}(\mu_i,\nu_j) \nonumber \\ = \frac{1}{36}
\begin{bmatrix} \nu_j^3 \\ \nu_j^2 \\ \nu_j \\ 1 \end{bmatrix}^T
\underbrace{\begin{bmatrix}
  -1 & 3 & -3 & 1\\
  3 & -6 & 3 & 0\\
  -3 & 0 & 3 & 0\\
  1 & 4 & 1 & 0
\end{bmatrix}}_{\text{IP}}
\begin{bmatrix}
  Q_{i,j} & Q_{i+1,j} & Q_{i+2,j} & Q_{i+3,j}\\
  Q_{i,j+1} & Q_{i+1,j+1} & Q_{i+2,j+1} & Q_{i+3,j+1}\\
  Q_{i,j+2} & Q_{i+1,j+2} & Q_{i+2,j+2} & Q_{i+3,j+2}\\
  Q_{i,j+3} & Q_{i+1,j+3} & Q_{i+2,j+3} & Q_{i+3,j+3}
\end{bmatrix}
\underbrace{\begin{bmatrix}
  -1 & 3 & -3 & 1\\
  3 & -6 & 0 & 4\\
  -3 & 3 & 3 & 1\\
  1 & 0 & 0 & 0
\end{bmatrix}}_{\text{IP}}
\begin{bmatrix} \mu_i^3 \\ \mu_i^2 \\ \mu_i \\ 1 \end{bmatrix}
\end{equation}
```

A reference for the calculations in this script can be found in Chapter 2 of
-  Quentin Agrapart & Alain Batailly (2020)
   Cubic and bicubic spline interpolation in Python. 
   [hal-03017566v2](https://hal.archives-ouvertes.fr/hal-03017566v2)
"""
function spline_interpolation(b_spline::BicubicBSpline, u, v)

  x  = b_spline.x
  y  = b_spline.y
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP

  i = max(1, min(searchsortedlast(x, u), length(x) - 1))
  j = max(1, min(searchsortedlast(y, v), length(y) - 1))

  my = (u - x[i])/h
  ny = (v - y[j])/h

  Q_temp = [Q[i, j:(j+3)] Q[(i+1), j:(j+3)] Q[(i+2), j:(j+3)] Q[(i+3), j:(j+3)]]

  c = 1/36 * [ny^3, ny^2, ny, 1]' * IP * Q_temp * IP' * [my^3, my^2, my, 1]
  
  return c
end

end # @muladd