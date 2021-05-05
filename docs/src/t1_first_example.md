# Tutorial 1: First Example
[![](https://mybinder.org/badge_logo.svg)](<unknown>/src/notebooks/t1_first_example.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](<unknown>/notebooks/t1_first_example.ipynb)

# Rational numbers

In julia rational numbers can be constructed with the `//` operator.
Lets define two rational numbers, `x` and `y`:

```julia
# Define variable x and y
x = 1//3
y = 2//5
```

```
2//5
```

When adding `x` and `y` together we obtain a new rational number:

```julia
z = x + y
x=range(-1, 1, length=50)
y=rand(50)
using Plots
plot(x, y)
```
![](3919954218.png)

Test for latex:
```math
\left\lbrace
\begin{aligned}
-∇\cdot\sigma(u) = 0 \ &\text{in} \ \Omega,\\
u = 0 \ &\text{on}\ \Gamma_{\rm G},\\
u_1 = \delta \ &\text{on}\ \Gamma_{\rm B},\\
\sigma(u)\cdot n = 0 \ &\text{on}\  \Gamma_{\rm N}.\\
\end{aligned}
\right.
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

