using Aqua

@testset "aqua quality checks" begin
    Aqua.test_ambiguities(FastIsostasy)
    Aqua.test_unbound_args(FastIsostasy)
    Aqua.test_undefined_exports(FastIsostasy)
    Aqua.test_project_extras(FastIsostasy)
    Aqua.test_stale_deps(FastIsostasy)
    Aqua.test_deps_compat(FastIsostasy)
    Aqua.test_piracies(FastIsostasy)
    Aqua.test_persistent_tasks(FastIsostasy)
end
