# fit_prony_series (fastisostasy-roadmap/burgers.md Phase 4): discretise the continuous
# Faul-Jackson absorption-band spectrum underlying I&C 2021's extended Burgers
# model into an N-branch Prony series for TransientCreepMantle.
#
# The load-bearing checks are (a) the N log-spaced bins exactly partition the
# band, so sum(Δⱼ) == relaxation_strength for any N (b) fit_error decreases
# monotonically as N grows, converging on the N ≈ 3-5 the roadmap expects to
# land within a few percent, and (c) BurgersMantle/ExtendedBurgersMantle wire
# the fit into TransientCreepMantle correctly.

using FastIsostasy
using Test

const FI = FastIsostasy

@testset "fit_prony_series (roadmap §6)" begin

    @testset "bin weights exactly partition relaxation_strength" begin
        for N in (1, 2, 3, 5, 8), α in (0.3, 0.5, 0.8)
            Δ, τ, _ = fit_prony_series(relaxation_strength = 1.2, alpha = α,
                tau_L = 1.0, tau_H = 100.0, nbranches = N)
            @test length(Δ) == length(τ) == N
            @test sum(Δ) ≈ 1.2
            @test all(>(0), Δ)
            @test all(τj -> 1.0 <= τj <= 100.0, τ)
            @test issorted(τ)
        end
    end

    @testset "fit_error vs N (nbranches-vs-cost study)" begin
        # α = 1/2 is the spectrum shape fastisostasy-roadmap/burgers.md §5 calls out for the
        # I&C 2021 comparison; τ_L, τ_H span two decades (years to a century),
        # matching the "years to centuries" range in §1's design table.
        errors = [fit_prony_series(relaxation_strength = 1.2, alpha = 0.5,
            tau_L = 1.0, tau_H = 100.0, nbranches = N)[3] for N in 1:8]

        @test issorted(errors, rev = true)  # strictly more branches, never worse
        @test errors[1] > 0.1               # N = 1 cannot resolve a 2-decade band
        @test errors[3] < 0.05              # N = 3 lands within a few percent...
        @test errors[5] < 0.01              # ...and N = 5 within about 1%,
        # ...matching the roadmap's "N ≈ 3-5 usually fits to within a few
        # percent" expectation. Empirically (this exact config): 16.6%, 5.6%,
        # 2.5%, 1.4%, 0.9%, 0.6%, 0.5%, 0.4% for N = 1..8.
    end

    @testset "guard rails" begin
        @test_throws ArgumentError fit_prony_series(relaxation_strength = 1.2,
            alpha = 0.0, tau_L = 1.0, tau_H = 100.0, nbranches = 4)
        @test_throws ArgumentError fit_prony_series(relaxation_strength = 1.2,
            alpha = 0.5, tau_L = 100.0, tau_H = 1.0, nbranches = 4)
        @test_throws ArgumentError fit_prony_series(relaxation_strength = -1.2,
            alpha = 0.5, tau_L = 1.0, tau_H = 100.0, nbranches = 4)
        @test_throws ArgumentError fit_prony_series(relaxation_strength = 1.2,
            alpha = 0.5, tau_L = 1.0, tau_H = 100.0, nbranches = 0)
    end

    @testset "BurgersMantle / ExtendedBurgersMantle wire the fit through" begin
        m1 = BurgersMantle(shearmodulus = 67e9, relaxation_strength = 1.2,
            kelvin_time = 7.14)
        @test m1 isa FI.TransientCreepMantle{Float64,1}
        @test FI.nbranches(m1) == 1

        m2 = ExtendedBurgersMantle(shearmodulus = 67e9, relaxation_strength = 1.2,
            alpha = 0.5, tau_L = 1.0, tau_H = 100.0, nbranches = 4)
        @test FI.nbranches(m2) == 4
        @test sum(m2.relaxation_strength) ≈ 1.2
        Δ_direct, τ_direct, _ = fit_prony_series(relaxation_strength = 1.2,
            alpha = 0.5, tau_L = 1.0, tau_H = 100.0, nbranches = 4)
        @test m2.relaxation_strength == Δ_direct
        @test m2.kelvin_time == τ_direct
    end
end
