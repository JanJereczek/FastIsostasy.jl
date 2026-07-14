#=
# Type tree

FastIsostasy is organized around a handful of small `abstract type` hierarchies
(one per concern: domains, mantle rheology, sea-level contributions, ice-loading
schemes, ...) rather than one big class tree. The snippet below walks the
module with [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)
and prints every such hierarchy, so that this page stays in sync with the
source instead of being maintained by hand.
=#

using FastIsostasy, AbstractTrees, InteractiveUtils

AbstractTrees.children(T::Type) = subtypes(T)

roots = filter(names(FastIsostasy; all = true)) do name
    isdefined(FastIsostasy, name) || return false
    obj = getfield(FastIsostasy, name)
    obj isa Type && isabstracttype(obj) && supertype(obj) === Any
end
roots = sort(unique(getfield.(Ref(FastIsostasy), roots)); by = string)

for root in roots
    print_tree(root)
end
