using Literate: Literate
using Test: @testset
import Pkg

# Create markdown and notebook files for `file`
function create_files(title, file, repo_src, pages_dir, notebooks_dir; folder = "")
    notebook_filename = first(splitext(file)) * ".ipynb"
    if !isempty(folder)
        notebook_filename = joinpath(folder, notebook_filename)
    end

    binder_logo = "https://mybinder.org/badge_logo.svg"
    nbviewer_logo = "https://img.shields.io/badge/render-nbviewer-f37726"
    raw_notebook_logo = "https://img.shields.io/badge/raw-notebook-4cc61e"

    notebook_path = "tutorials/notebooks/$notebook_filename"
    binder_url = "https://mybinder.org/v2/gh/trixi-framework/Trixi.jl/tutorial_notebooks?filepath=$notebook_path"
    nbviewer_url = "https://nbviewer.jupyter.org/github/trixi-framework/Trixi.jl/blob/tutorial_notebooks/$notebook_path"
    raw_notebook_url = "https://raw.githubusercontent.com/trixi-framework/Trixi.jl/tutorial_notebooks/$notebook_path"

    binder_badge = "# [![]($binder_logo)]($binder_url)"
    nbviewer_badge = "# [![]($nbviewer_logo)]($nbviewer_url)"
    raw_notebook_badge = "# [![]($raw_notebook_logo)]($raw_notebook_url)"

    # Generate notebook file
    function preprocess_notebook(content)
        warning = "# **Note:** To improve responsiveness via caching, the notebooks are updated only once a week. They are only
        # available for the latest stable release of Trixi.jl at the time of caching.\n\n"
        return string("# # $title\n\n", warning, content)
    end
    Literate.notebook(joinpath(repo_src, folder, file), joinpath(notebooks_dir, folder);
                      execute = false, preprocess = preprocess_notebook, credit = false)

    # Generate markdown file
    function preprocess_docs(content)
        return string("# # [$title](@id $(splitext(file)[1]))\n $binder_badge\n $nbviewer_badge\n $raw_notebook_badge\n\n",
                      content)
    end
    Literate.markdown(joinpath(repo_src, folder, file), joinpath(pages_dir, folder);
                      preprocess = preprocess_docs,)
end

# Create tutorials with Literate.jl
function create_tutorials(files)
    repo_src = joinpath(@__DIR__, "src", "files")

    pages_dir = joinpath(@__DIR__, "..", "src", "tutorials")
    notebooks_dir = joinpath(pages_dir, "notebooks")

    Sys.rm(pages_dir; recursive = true, force = true)

    Sys.rm("out"; recursive = true, force = true)

    # Run tests on all tutorial files
    @testset "TrixiTutorials" begin
        for (i, (title, filename)) in enumerate(files)
            # Evaluate each tutorial in its own module to avoid leaking of
            # function/variable names, polluting the namespace of later tutorials
            # by stuff defined in earlier tutorials.
            if filename isa Vector # Several files of one topic
                for j in eachindex(filename)
                    mod = gensym(filename[j][2][2])
                    @testset "$(filename[j][2][2])" begin
                        @eval module $mod
                        include(joinpath($repo_src, $(filename[j][2][1]),
                                         $(filename[j][2][2])))
                        end
                    end
                end
            else # Single files
                mod = gensym(title)
                @testset "$title" begin
                    @eval module $mod
                    include(joinpath($repo_src, $filename))
                    end
                end
            end
        end
    end

    # Generate markdown file for introduction page
    # Preprocessing introduction file: Generate consecutive tutorial numbers by replacing
    # each occurrence of `{index}` with an integer incremented by 1, starting at 1.
    function preprocess_introduction(content)
        counter = 1
        while occursin("{index}", content)
            content = replace(content, "{index}" => "$counter", count = 1)
            counter += 1
        end
        return content
    end
    Literate.markdown(joinpath(repo_src, "index.jl"), pages_dir; name = "introduction",
                      preprocess = preprocess_introduction)
    # Navigation system for makedocs
    pages = Any["Introduction" => "tutorials/introduction.md"]

    # Create markdown and notebook files for tutorials
    for (i, (title, filename)) in enumerate(files)
        # Several files of one topic are created separately and pushed to `pages` together.
        if filename isa Vector
            vector = []
            for j in eachindex(filename)
                create_files("$i.$j: $title: $(filename[j][1])", filename[j][2][2],
                             repo_src,
                             pages_dir, notebooks_dir; folder = filename[j][2][1])

                path = "$(filename[j][2][1])/$(splitext(filename[j][2][2])[1]).md"
                push!(vector, "$i.$j $(filename[j][1])" => "tutorials/$path")
            end
            # Add to navigation menu
            push!(pages, ("$i $title" => vector))
        else # Single files
            create_files("$i: $title", filename, repo_src, pages_dir, notebooks_dir)
            # Add to navigation menu
            path = first(splitext(filename)) * ".md"
            push!(pages, ("$i $title" => "tutorials/$path"))
        end
    end

    return pages
end
