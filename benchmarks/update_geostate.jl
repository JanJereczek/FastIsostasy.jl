include("helpers.jl")

fip = minimal_fip()
u = copy(fip.out.u[1])
dudt = copy(fip.out.dudt[1])
t = years2seconds(200.0)
# update_loadcolumns!(fip, fip.tools.Hice(t))
# columnanom_load!(fip)
# update_elasticresponse!(fip)
# columnanom_litho!(fip)
# update_geoid!(fip)

# update_sealevel!(fip)

@profview for i in 1:100
    update_diagnostics!(dudt, u, fip, t)
end

@btime update_diagnostics!($dudt, $u, $fip, $t)
# V_antarctica = 26e6 * (1e3)^3
# nV = 1000
# dV = V_antarctica / nV
# for i in 1:nV
#     fip.now.osc(-dV)
# end