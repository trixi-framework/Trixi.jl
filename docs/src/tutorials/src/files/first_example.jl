# # Rational numbers
#
# In julia rational numbers can be constructed with the `//` operator.
# Lets define two rational numbers, `x` and `y`:

## Define variable x and y
x = 1//3
y = 2//5

# When adding `x` and `y` together we obtain a new rational number:

z = x + y
x=range(-1, 1, length=50)
y=rand(50)
using Plots
plot(x, y)
# Test for latex:
# ```math
# \left\lbrace
# \begin{aligned}
# -∇\cdot\sigma(u) = 0 \ &\text{in} \ \Omega,\\
# u = 0 \ &\text{on}\ \Gamma_{\rm G},\\
# u_1 = \delta \ &\text{on}\ \Gamma_{\rm B},\\
# \sigma(u)\cdot n = 0 \ &\text{on}\  \Gamma_{\rm N}.\\
# \end{aligned}
# \right.
# ```
