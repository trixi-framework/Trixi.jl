# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

##############################################
### One dimensional B-spline interpolation ###
##############################################

# Linear B-spline interpolation
@doc raw"""
    spline_interpolation(b_spline::LinearBSpline, t)

The inputs are the [`LinearBSpline`](@ref) object and a variable `t` at which the spline 
will be evaluated.

The parameter `i` gives us the patch in which the variable `t` is located.
This parameter will also be used to get the correct control points from `Q`.
(A patch is the area between two consecutive `x` values. `h`)

`kappa` is  an interim variable which maps `t` to the interval ``[0,1]``
for further calculations.

To evaluate the spline at `t`, we have to calculate the following:
```math
\begin{equation}
c_{i,1}(\kappa_i) = 
		\begin{bmatrix}
			\kappa_i\\ 1
		\end{bmatrix}^T
		\underbrace{\begin{bmatrix}
			-1 & 1\\1 & 0
		\end{bmatrix}}_{\text{IP}}
		\begin{bmatrix}
			Q_i\\Q_{i+1}
\end{equation}
```

A reference for the calculations in this script can be found in Chapter 1 of
-  Quentin Agrapart & Alain Batailly (2020)
   Cubic and bicubic spline interpolation in Python. 
   [hal-03017566v2](https://hal.archives-ouvertes.fr/hal-03017566v2)
"""
function spline_interpolation(b_spline::LinearBSpline, t)

  x  = b_spline.x
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP

  i = max(1, min(searchsortedlast(x, t), length(x)-1))

  kappa_i = (t - x[i])/h

  c_i1 = [kappa_i, 1]' * IP * Q[i:(i+1)]

  return c_i1  
end

# Cubic B-spline interpolation
@doc raw"""
    spline_interpolation(b_spline::CubicBSpline, t)

The inputs are the [`CubicBSpline`](@ref) object and a variable `t` at which the spline 
will be evaluated.

The parameter `i` gives us the patch in which the variable `t` is located.
This parameter will also be used to get the correct control points from `Q`.
(A patch is the area between two consecutive `x` values)

`kappa` is  an interim variable which maps `t` to the interval ``[0,1]``
for further calculations.

To evaluate the spline at `t`, we have to calculate the following:
```math
\begin{equation}
c_{i,3}\left(\kappa_i(t) \right) = \frac{1}{6} 
		\begin{bmatrix}
			\kappa_i(t)^3\\ \kappa_i(t)^2\\ \kappa_i(t) \\1
		\end{bmatrix}^T
		\underbrace{\begin{bmatrix}
			-1 & 3 & -3 & 1\\
			3 & -6 & 3 & 0\\
			-3 & 0 & 3 & 0\\
			1 & 4 & 1 & 0
		\end{bmatrix}}_{\text{IP}}
		\begin{bmatrix}
			Q_{i,\text{free}}\\ Q_{i+1,\text{free}}\\ Q_{i+2,\text{free}}\\ Q_{i+3,\text{free}}
		\end{bmatrix}
\end{equation}
```

A reference for the calculations in this script can be found in Chapter 1 of
-  Quentin Agrapart & Alain Batailly (2020)
   Cubic and bicubic spline interpolation in Python. 
   [hal-03017566v2](https://hal.archives-ouvertes.fr/hal-03017566v2)
"""
function spline_interpolation(b_spline::CubicBSpline, t)

  x  = b_spline.x
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP

  i = max(1, min(searchsortedlast(x, t), length(x) - 1))

  kappa_i = (t - x[i])/h

  c_i3 = 1/6 * [kappa_i^3, kappa_i^2, kappa_i, 1]' * IP * Q[i:(i+3)]

  return c_i3
end

end # @muladd