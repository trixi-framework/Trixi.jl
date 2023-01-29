```@meta
EditURL = "<unknown>/src/Getting_started_with_Trixi.jl"
```

# **Getting started with Trixi.jl**

**Trixi.jl** is a numerical simulation framework for hyporbolic conservation laws
written in [`Julia`](https://julialang.org/).
That means for working with Trixi, Julia have to be installed on a PC.

## **1. Julia installation**

Trixi works with the current stable release Julia v.1.8.5.
The most fully explaind installation process can be found in this
[`Julia installation instruction`](https://julialang.org/downloads/platform/).
But you can follow also our short installation instruction.

### **1.1. Windows**

- Download Julia [`installer`](https://julialang.org/downloads/) for Windows. Make sure
that you chose the right version of installer (64-bit or 32-bit) according to your computer.
- Open the downloaded installer.
- Paste an installation directory path or find it using a file manager (select *Browse*).
- Select *Next*.
- Check the *Add Julia to PATH* to add Julia to Environment Variables.
  This makes possible to run Julia using Terminal from any directory only typing *julia*.
- Select *Next*, then Julia will be insalled.

Now you can verify, that Julia is installed:
- Type *Win+R* on a keyboard.
- Enter *cmd* in opened window + *Enter*.
- Enter in a terminal *julia* + *Enter*.

Then Julia will be invoked. To close Julia enter *exit()* + *Enter*.

### **1.2. Linux**

- Open a terminal and navigate (using *cd*) to a directory, where you want to save Julia.
Or you can open file manager, find this directory, right-click inside and
choose *Open Terminal Here*.
- To install Julia execute the following commands in the Terminal:
````
wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
tar zxvf julia-1.8.5-linux-x86_64.tar.gz
````
Now you can verify that Julia is installed entering *julia* command in the Terminal.

Then Julia will be invoked. To close Julia enter *exit()* + *Enter*.

## **2. Trixi installation**

### **2.1. For Users**

If you are planning to use Trixi for work or study without making any changes in Trixi,
then you can follow this instruction. If you are planning to develop Trixi, then follow
topic **Trixi installation for developers**.

Trixi and its related tools are registered Julia packages. So installation of them is
running inside Julia. To appropriate work of Trixi you need to install
[`Trixi`](https://github.com/trixi-framework/Trixi.jl),
[`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl),
[`OrdinaryDiffEq`](https://github.com/SciML/OrdinaryDiffEq.jl) and
[`Plots`](https://github.com/JuliaPlots/Plots.jl).

- Open a Terminal: type *Win + R* and enter *cmd*.
- Invoke Julia executing *julia*.
- Execute following commands:

````@example Getting_started_with_Trixi
import Pkg
Pkg.add(["Trixi", "Trixi2Vtk", "OrdinaryDiffEq", "Plots"])
````

Now you have installed all this packages.
[`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl) is a visualization tool,
[`OrdinaryDiffEq`](https://github.com/SciML/OrdinaryDiffEq.jl) provides time integration schemes
used by Trixi and [`Plots`](https://github.com/JuliaPlots/Plots.jl) can be used to directly
visualize Trixi's results from the Julia REPL.

### **2.2. For Developers**

If you plan on editing Trixi itself, you can download Trixi locally and run it from within the
cloned directory.

#### **2.2.1. Windows**

If you are using Windows OS, you can clone Trixi directory using a Github Desktop.
- If you haven't any github account yet, you have to create it on the
[`Github website`](https://github.com/join).
- Download and install [`Github Desktop`](https://desktop.github.com/) and then login into
your account.
- Open an installed Github Desktop, type *Ctrl+Shoft+O*.
- In opened window paste *trixi-framework/Trixi.jl* and choose path to a folder, where you want
to save Trixi. Then click *Clone* and Trixi will be cloned to PC.

Now you cloned Trixi and only need to add Trixi packages to Julia.
- Open Terminal using *Win+R* and *cmd*. Navigate to the folder with cloned Trixi using *cd*.
- Start Julia with the ````--project```` flag set to your local Trixi clone, e.g.,
````
 julia --project=@.
````
- Run following commands in Julia REPL:

````@example Getting_started_with_Trixi
import Pkg; Pkg.instantiate()
Pkg.add(["Trixi2Vtk", "Plots", "OrdinaryDiffEq"])
````

Now you already installed Trixi from your local clone. Note that if you installed Trixi this way,
you always have to start Julia with the ````--project```` flag set to your local Trixi clone, e.g.,
````
julia --project=@.
````

#### **2.2.2. Linux**

You can download Trixi locally and run it from within the cloned directory this way:
````
git clone git@github.com:trixi-framework/Trixi.jl.git
cd Trixi.jl
julia --project=@. -e 'import Pkg; Pkg.instantiate()'
julia -e 'import Pkg; Pkg.add(["Trixi2Vtk", "Plots"])'
julia -e 'import Pkg; Pkg.add("OrdinaryDiffEq")'
````
Note that if you installed Trixi this way,
you always have to start Julia with the ````--project```` flag set to your local Trixi clone, e.g.,
````
julia --project=@.
````

## **3. Usage**

### **3.1. Files execution**

Trixi has a big set of
[`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples), that can be taken
as basis for your future investigations.

Now execute one of them using *trixi_include(...)* function. *trixi_include(...)* expects
a single string argument with the path to a Trixi elixir, i.e., a text file containing Julia
code necessary to set up and run a simulation. *default_example()* returns the path to an example
elixir with a short, two-dimensional problem setup. *plot(sol)* builds a graphical representation
of the solution.

Invoke Julia in terminal. (Open Terminal: *Win+R* and enter *cmd*, invoke Julia in terminal: *julia*).
And execute following code.

````@example Getting_started_with_Trixi
using Trixi
trixi_include(default_example())
using Plots
plot(sol)
````

To obtain list of all example elixirs packaged with Trixi execute *get_examples()*. This will
return pathes to all examples.

````@example Getting_started_with_Trixi
get_examples()
````

Editing the Trixi examples are the best way to start your first own investigation using Trixi.

### **3.2. Files edditing**

#### **3.2.1 For Users**

To edit example files you have to download them. Let's have a look how to download default
example file from [`Trixi github`](https://github.com/trixi-framework/Trixi.jl).

- All examples are located inside the
[`example`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder.
- Navigate to the file
[`elixir_advection_basic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/structured_2d_dgsem/elixir_advection_basic.jl).
- Click the *Raw* button on right side of the webpage.
- On any place of newly opened webpage right-click and choose *Save as*.
- Choose folder and erase *.txt* from the file name. Save the file.

Now you can change something in this file. For example change the initial conditions.

- Open the file you downloaded. And go to the 10th line with following code:
````
advection_velocity = (0.2, -0.7)
````
- Change values from *(0.2, -0.7)* to *(0.1, 0.1)*.
- Execute following code one more time, but instead of *path_to_file* paste the path to the
elixir_advection_basic.jl file from current folder, for example *"./elixir_advection_basic.jl"*.
````
using Trixi
trixi_include(path_to_file)
using Plots
plot(sol)
````
You will obtain new plot with shifted dark and light lines.

#### **3.2.2. For Developers**

Like a developer you can change example files inside a cloned repository.
To find a path to the *default_example* run following command in julia REPL:

````@example Getting_started_with_Trixi
default_example()
````

Open this file and go to the 10th line with following code:
````
advection_velocity = (0.2, -0.7)
````
Change values from *(0.2, -0.7)* to *(0.1, 0.1)*.
Execute following code one more time:
````
using Trixi
trixi_include(default_example())
using Plots
plot(sol)
````
You will obtain new plot with shifted dark and light lines.

Now you are able to download, edit and execute Trixi code.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

