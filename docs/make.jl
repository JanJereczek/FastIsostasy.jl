# Standard stuff
cd(@__DIR__)
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using CairoMakie, Documenter, Literate
using DocumenterTools: Themes
using DocumenterCitations
ENV["JULIA_DEBUG"] = "Documenter"

# Packages specific to these docs
push!(LOAD_PATH, "../")
using FastIsostasy

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

Literate.markdown("src/examples/tutorial.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/glacialcylce.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/inversion.jl", "src/examples"; credit = false)

example_pages = ["examples/glacialcylce.md", "examples/inversion.md"]
ref_pages = ["APIref.md", "fortran.md", "references.md"]
# %% Build docs
PAGES = [
    "index.md",
    "introGIA.md",
    "examples/tutorial.md",
    "Examples" => example_pages,
    "References" => ref_pages,
]

include("style.jl")

makedocs(
    modules = [FastIsostasy],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 2,
        ),
    sitename = "FastIsostasy.jl",
    authors = "Jan Swierczek-Jereczek",
    pages = PAGES,
    doctest = CI,
    draft = false,
    plugins=[bib],
    checkdocs = :none,
    warnonly = true,
)

deploydocs(;
    repo="github.com/JanJereczek/FastIsostasy.jl",
)