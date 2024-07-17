config = InversionConfig()
data = InversionData()
reconstruct_params!(fip, x)
prob = InversionProblem(config, data, reconstruct_params)