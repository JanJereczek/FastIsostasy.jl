# Standard stuff
using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using CairoMakie, Documenter, Literate
using DocumenterTools: Themes
using DocumenterCitations
import Bibliography
ENV["JULIA_DEBUG"] = "Documenter"

# Packages specific to these docs
using FastIsostasy

# DocumenterCitations doesn't expose Bibliography's `check` keyword, so entries
# with missing BibTeX fields (e.g. `journal` on a preprint) make the build
# error out. Relax this to a warning instead of failing `makedocs`.
Bibliography.import_bibtex(bibfile::AbstractString) = Bibliography.import_bibtex(bibfile; check = :warn)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "fastiso.bib");
    style=:authoryear
)

Literate.markdown("src/examples/benchmark_analytic.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/benchmark_1D.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/benchmark_3D.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/glacialcycle.jl", "src/examples"; credit = false)

Literate.markdown("src/examples/elra.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/green_functions.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/transient_creep.jl", "src/examples"; credit = false)

Literate.markdown("src/examples/coupling.jl", "src/examples"; credit = false)

Literate.markdown("src/examples/inverse_calibration.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/inverse_ice_history.jl", "src/examples"; credit = false)
Literate.markdown("src/examples/inverse_fullfield.jl", "src/examples"; credit = false)

Literate.markdown("src/examples/benchmark_realfft.jl", "src/examples"; credit = false)
Literate.markdown("src/treestructure.jl", "src"; credit = false)

maxwell_earth = [
    "examples/benchmark_analytic.md",
    "examples/benchmark_1D.md",
    "examples/benchmark_3D.md",
    "examples/glacialcycle.md",
]

alternative_models = [
    "examples/elra.md",
    "examples/green_functions.md",
    "examples/transient_creep.md",
]

inverse_problems = [
    "examples/inverse_calibration.md",
    "examples/inverse_ice_history.md",
    "examples/inverse_fullfield.md",
]

advanced_topics = [
    "integrators.md",
    "examples/benchmark_realfft.md",
    "inversion_ad_activity_map.md",
    "transient_creep_derivation.md",
    "treestructure.md",
]

ref_pages = [
    "API_public.md",
    "API_public_ext.md",
    "API_private.md",
    "fortran.md",
    "publications.md",
    "references.md",
]
# %% Build docs
PAGES = [
    "index.md",
    "introGIA.md",
    "Forward runs" => [
        "Maxwell Earth" => maxwell_earth,
        "Alternative Models" => alternative_models,
        "examples/coupling.md",
    ],
    "Inverse problems" => inverse_problems,
    "Advanced Topics" => advanced_topics,
    "API & References" => ref_pages,
]

include("style.jl")

makedocs(
    modules = [FastIsostasy],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 1,
        ),
    sitename = "FastIsostasy.jl",
    authors = "Jan Swierczek-Jereczek",
    pages = PAGES,
    doctest = CI,
    draft = false,
    plugins = [bib],
    checkdocs = :none,
    warnonly = true,
)

deploydocs(;
    repo="github.com/JanJereczek/FastIsostasy.jl",
)