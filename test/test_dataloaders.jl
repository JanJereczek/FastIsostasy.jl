@testset "data loaders" begin
    datasets = ["Antarctic3RegionMask", "OceanSurfaceFunctionETOPO2022", "BedMachine3",
        "ICE6G_D", "Wiens2022", "Lithothickness_Pan2022", "Viscosity_Pan2022"]
    for dataset in datasets
        dims, field, itp = load_dataset(dataset)
        @test (dims isa Tuple || dims isa AbstractVector)
        @test field isa AbstractArray
    end
end