# Regression tests for src/material.jl's `channel_scaling` (roadmap
# stabilise_dt.md §3.1). The frequency-domain Cathles channel-flow correction
# used to build C = cosh(x), S = sinh(x) with x = channel_thickness * kappa
# directly. For a thick channel on a fine grid x reaches ~118 at this package's
# default geometry, where cosh/sinh overflow to Inf in Float32 (arg threshold
# ~88.7) — and to Inf in Float64 by ~354 — and the subsequent num/denum degrades
# Inf/Inf to NaN, silently corrupting the physics of *any* Float32/GPU
# laterally-variable-lithosphere run. The overflow-safe rewrite must stay finite
# and reproduce the (non-overflowing) reference exactly.

using FastIsostasy
using Test

@testset "material" begin

    @testset "channel_scaling: no Float32 NaN at the baseline (§3.1 regression)" begin
        # A Float32 SolidEarth at the roadmap baseline used to produce NaN in
        # ~14% (36053/262144) of pseudodiff_scaling grid points via the default
        # FreqDomainViscosityLumping path; it must now be everywhere finite.
        W, n = 3f6, 9
        domain = RegionalDomain(W, n)
        @test eltype(domain) == Float32

        # Sanity: this config genuinely exercises the overflow branch — the
        # argument channel_thickness * kappa must exceed Float32's cosh overflow
        # threshold, else the test would pass trivially on a benign config.
        channel_thickness = 400f3 - 88f3
        maxarg = channel_thickness * maximum(domain.pseudodiff)
        @test maxarg > log(floatmax(Float32))        # ~88.7 (here ~118)

        se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
            layer_boundaries = [88f3, 400f3], layer_viscosities = [1f18, 1f21])
        ps = se.pseudodiff_scaling
        @test eltype(ps) == Float32
        @test all(isfinite, ps)
        @test !any(isnan, ps)
    end

    @testset "channel_scaling: limits and boundedness" begin
        # Only eltype(domain) is consumed by channel_scaling; a small grid is fine.
        domain = RegionalDomain(3e6, 5)
        nu, h = 0.37, 312e3

        # x -> 0: no correction at the DC mode (cf. §4).
        @test isapprox(FastIsostasy.channel_scaling(domain, 0.0, h, nu), 1.0; atol = 1e-8)

        # x -> Inf: -> visc_ratio, and finite despite cosh(x) overflowing.
        big = FastIsostasy.channel_scaling(domain, 118.0 / h, h, nu)   # arg ~118
        @test isfinite(big)
        @test isapprox(big, nu; rtol = 1e-6)

        # Float32 at the same overflowing argument stays finite (cosh(118f0)==Inf).
        domain32 = RegionalDomain(3f6, 5)
        big32 = FastIsostasy.channel_scaling(domain32, 118f0 / 312f3, 312f3, 0.37f0)
        @test isfinite(big32)
        @test isapprox(big32, 0.37f0; rtol = 1f-4)
    end

end
