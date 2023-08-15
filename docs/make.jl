# Standard stuff
cd(@__DIR__)
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using CairoMakie
using Documenter
using DocumenterTools: Themes
ENV["JULIA_DEBUG"] = "Documenter"
# Packages specific to these docs
push!(LOAD_PATH, "../")
using FastIsostasy

# %% Build docs
PAGES = [
    "index.md",
    "introGIA.md",
    "examples.md",
    "APIref.md",
]

include("style.jl")

makedocs(
    modules = [FastIsostasy],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            # "assets/logo-dark.ico",
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 2,
        ),
    sitename = "FastIsostasy.jl",
    authors = "Jan Swierczek-Jereczek",
    pages = PAGES,
    doctest = CI,
    draft = true,
)

deploydocs(;
    repo="github.com/JanJereczek/FastIsostasy.jl",
)