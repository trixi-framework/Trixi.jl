# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#################################################
### Helpfunctions for 1D spline interpolation ###
#################################################

# Sorting the inputs so that the x values are ascending
"""
    sort_data(x::Vector,y::Vector)

Sorts the input vectors `x` and `y` so that `x` is in ascending order and the `y` values still
correspond to the `x` values.
"""
function sort_data(x::Vector,y::Vector)
      
  original_data = hcat(x,y)
  sorted_data = original_data[sortperm(original_data[:,1]), :]
  
  x_sorted = sorted_data[:,1]
  y_sorted = sorted_data[:,2]
  
  return x_sorted,y_sorted
end

# Spline smoothing
@doc raw"""
    spline_smoothing(lambda::Number, h::Number, y::Vector)

The inputs to this function are:
- `lambda`: a smoothing factor which specifies the degree of the smoothing that should take place
- `h`: the step size of a patch (A patch is the area between two consecutive `x` values)
- `y`: the `y` values to be smoothed

The goal is to find a new interpolation values ``\hat{y}`` for ``y``, so that for given ``\lambda``, 
the following equation is minimized:
```math
\begin{equation}
		\text{PSS} = \sum_{i = 1}^{n} \left( y_i - \underbrace{S(t_i)}_{=\hat{y}_i} \right)^2 
    + \lambda \int_{x_1}^{x_n} (S''(t))^2 dt,
	\end{equation}
```
where ``S(t)`` is a cubic spline function. 
``\hat{y}`` is determined as follows:
```math
\begin{equation}
\hat{y} = (I+\lambda K)^{-1} y
\end{equation}
```
where ``I`` is the ``n \times n`` identity matrix and ``K = \Delta^T W^{-1} \Delta`` with
```math
\begin{equation}
\Delta = \begin{pmatrix}
1/h & -2/h & 1/h & ... & 0\\
0 & \ddots & \ddots & \ddots & 0\\
0 & ... & 1/h & -2/h & 1/h
\end{pmatrix} \in \mathbb{R}^{(n-2) \times n}
\end{equation}
```
and
```math
\begin{equation}
W = \begin{pmatrix}
2/3 h & 1/6 h & 0 & ... & 0\\
1/6 h & 2/3 h & 1/6 h & ... & 0\\
0 & \ddots & \ddots & \ddots & 0\\
0 & ... & 0 & 2/3 h & 1/6 h
\end{pmatrix} \in \mathbb{R}^{n \times n}
\end{equation}
```

- Germán Rodríguez (2001)
  Smoothing and non-parametric regression
  (https://data.princeton.edu/eco572/smoothing.pdf)
"""
function spline_smoothing(lambda::Number, h::Number, y::Vector)

  n = length(y)

  h_vec = repeat([h], n-2)

  Delta_ii   =  1 ./ h_vec
  Delta_iip1 = -2 ./ h_vec
  Delta_iip2 =  1 ./ h_vec
  
  Delta             =  zeros(n-2, n)
  Delta[:, 1:(n-2)] =  diagm(Delta_ii)
  Delta[:, 2:(n-1)] += diagm(Delta_iip1)
  Delta[:, 3: n   ] += diagm(Delta_iip2)

  W_im1i =  h_vec[1:end-1] ./ 6
  W_ii   = (2*h_vec) ./ 3
  W      = SymTridiagonal(W_ii, W_im1i)

  K = transpose(Delta) * inv(W) * Delta

  return inv(diagm(ones(n)) + lambda*K) * y
end

end # @muladd