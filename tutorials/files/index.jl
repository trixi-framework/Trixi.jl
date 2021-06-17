#src # Welcome to the tutorials of Trixi.jl

# Working with somebody else's code often is hard in the beginning. To simplify the start looking into an example might help. It is even more helpful if you can run the example by yourself and see what single lines or some commands will do.
# So, in this tutorial section you have the possibility to do exactly this.
# There are three ways to use these tutorials:
# - If you have not much time or just want to look quickly in the example, you can use the documentation page. 
# - For a deeper look there is a not interactive way with `nbviewer` where the code is already executed.
# - To take full advantage you can use the interactive notebook version of the tutorials where you can run the code by yourself. To get there just click on the `binder` logo within the corresponding tutorial.

M = rand(3, 3)
size(M)

#-
size(M) == (3, 3)