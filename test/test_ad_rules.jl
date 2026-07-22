# Phase-2 foundational AD rules: the forward-mode planned-FFT `mul!` rule and the
# `t_computation!` inactive rule (roadmap §4). Validates the custom `EnzymeRules`
# in `FastIsostasyEnzymeExt` in isolation, before the full `gradient!` milestone.
#
# The differentiated functions are top-level and non-capturing; the FFT plans and
# the timer are passed as explicit `Const` arguments (a captured mutable object
# would trip Enzyme's "argument cannot be proven readonly" analysis).

using Test
using FastIsostasy
using FFTW
using LinearAlgebra: mul!
using Enzyme

# --- differentiated model functions (no captured state) ----------------------

# Real vector → complex fft → complex ifft → scalar (the update_dudt! pattern).
function _loss_cfft(x, n, pfft, pifft)
    X = complex.(reshape(x, n, n))
    Y = similar(X); mul!(Y, pfft, X)     # forward plan rule
    Z = similar(Y); mul!(Z, pifft, Y)    # forward plan rule
    return sum(abs2, real.(Z))
end

# Real vector → rfft → irfft → scalar (ConvolutionPlan / RealFFTBackend pattern).
function _loss_rfft(x, n, prfft, pirfft)
    X = reshape(x, n, n)
    Y = zeros(ComplexF64, n ÷ 2 + 1, n); mul!(Y, prfft, X)
    Z = zeros(n, n); mul!(Z, pirfft, Y)
    return sum(abs2, Z)
end

# Touches the wall-clock timer, which must contribute nothing to the derivative.
function _loss_timer(x, timer)
    FastIsostasy.t_computation!(timer)
    return x^3
end

# Central finite-difference directional derivative of `f(x)` along `dx`.
fd_dir(f, x, dx; ε = 1e-6) = (f(x .+ ε .* dx) - f(x .- ε .* dx)) / (2ε)

@testset "Phase-2 AD rules" begin

    @testset "forward plan mul! rule (complex fft/ifft)" begin
        n = 8
        pfft = plan_fft(zeros(ComplexF64, n, n))
        pifft = plan_ifft(zeros(ComplexF64, n, n))
        f(x) = _loss_cfft(x, n, pfft, pifft)

        x = randn(n * n); dx = randn(n * n)
        d_enzyme = only(Enzyme.autodiff(Forward, _loss_cfft,
            Duplicated(x, dx), Const(n), Const(pfft), Const(pifft)))
        @test isapprox(d_enzyme, fd_dir(f, x, dx); rtol = 1e-5)

        # Full gradient, component by component, vs finite differences.
        gθ = map(1:length(x)) do i
            e = zeros(length(x)); e[i] = 1.0
            only(Enzyme.autodiff(Forward, _loss_cfft,
                Duplicated(x, e), Const(n), Const(pfft), Const(pifft)))
        end
        gθ_fd = map(i -> fd_dir(f, x, (e = zeros(length(x)); e[i] = 1.0; e)), 1:length(x))
        @test isapprox(gθ, gθ_fd; rtol = 1e-5)
    end

    @testset "reverse plan mul! rule (complex fft/ifft)" begin
        n = 6
        pfft = plan_fft(zeros(ComplexF64, n, n))
        ## normalized inverse plan (the differentiated-path `pifft!`)
        pifft = FastIsostasy.normalize_plan(plan_ifft(zeros(ComplexF64, n, n)))
        f(x) = _loss_cfft(x, n, pfft, pifft)

        x = randn(n * n)
        ## reverse-mode gradient
        g = zeros(n * n)
        Enzyme.autodiff(Reverse, _loss_cfft, Active,
            Duplicated(x, g), Const(n), Const(pfft), Const(pifft))

        ## vs component-wise finite differences
        g_fd = map(1:length(x)) do i
            e = zeros(length(x)); e[i] = 1.0
            fd_dir(f, x, e)
        end
        @test isapprox(g, g_fd; rtol = 1e-5)

        ## and vs forward-mode AD (should agree to ~machine precision)
        g_fwd = map(1:length(x)) do i
            e = zeros(length(x)); e[i] = 1.0
            only(Enzyme.autodiff(Forward, _loss_cfft, Duplicated(x, e),
                Const(n), Const(pfft), Const(pifft)))
        end
        @test isapprox(g, g_fwd; rtol = 1e-8)
    end

    @testset "forward plan mul! rule (rfft/irfft)" begin
        n = 8
        prfft = plan_rfft(zeros(n, n))
        pirfft = plan_irfft(zeros(ComplexF64, n ÷ 2 + 1, n), n)
        f(x) = _loss_rfft(x, n, prfft, pirfft)

        x = randn(n * n); dx = randn(n * n)
        d_enzyme = only(Enzyme.autodiff(Forward, _loss_rfft,
            Duplicated(x, dx), Const(n), Const(prfft), Const(pirfft)))
        @test isapprox(d_enzyme, fd_dir(f, x, dx); rtol = 1e-5)
    end

    @testset "reverse plan mul! rule (rfft/irfft)" begin
        ## even and odd n: Nyquist row exists only for even n, and the rfft/brfft
        ## adjoints scale the interior frequencies differently around it.
        for n in (6, 7)
            m = n ÷ 2 + 1
            prfft = plan_rfft(zeros(n, n))
            pirfft = FastIsostasy.normalize_plan(plan_irfft(zeros(ComplexF64, m, n), n))
            f(x) = _loss_rfft(x, n, prfft, pirfft)

            x = randn(n * n)
            g = zeros(n * n)
            ## real transforms reuse buffers ⇒ runtime activity, as in `gradient!`
            Enzyme.autodiff(Enzyme.set_runtime_activity(Reverse), _loss_rfft, Active,
                Duplicated(x, g), Const(n), Const(prfft), Const(pirfft))

            g_fd = map(1:length(x)) do i
                e = zeros(length(x)); e[i] = 1.0
                fd_dir(f, x, e)
            end
            @test isapprox(g, g_fd; rtol = 1e-5)
        end
    end

    @testset "t_computation! inactive rule" begin
        timer = FastIsostasy.Timer((0.0, 10.0); T = Float64)
        timer.t = 5.0    # past t_span[1] ⇒ the push! branch is live if not inactive
        d = only(Enzyme.autodiff(Forward, _loss_timer,
            Duplicated(2.0, 1.0), Const(timer)))
        @test isapprox(d, 3 * 2.0^2; rtol = 1e-10)   # 3x², timer ignored
    end
end
