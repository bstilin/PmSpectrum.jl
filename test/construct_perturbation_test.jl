# ------------------------------------------
# Tests for ConstructPerturbationSolution.jl
# ------------------------------------------


using Test
using SpecialFunctions: loggamma, gamma
using PmSpectrum.Utils: exp_times_upper_gamma, construct_perturbation_solution
# (Double64 comes from `using PmSpectrum` in runtests.jl — PmSpectrum re-exports it.)


# High-precision reference for E(s,z) = exp(z)·Γ(s,z), via MPFR's incomplete
# gamma (`loggamma(a, x)` has a BigFloat method). Working in the log domain
# avoids under/overflow at large z.
function E_bigfloat_ref(s, z; prec=512)
    setprecision(BigFloat, prec) do
        sb = BigFloat(s)
        zb = BigFloat(z)
        zb >= 0 || throw(ArgumentError("Require z >= 0"))
        return exp(zb + loggamma(sb, zb))
    end
end

relerr(x, xref) = abs(BigFloat(x) - xref) / abs(xref)


######################################################
# exp_times_upper_gamma — closed forms for integer s
#
# For integer s, the upper incomplete gamma has the exact closed form:
#   Γ(1, z) = exp(-z)                      → exp(z)·Γ(1,z) = 1
#   Γ(2, z) = (1 + z) exp(-z)              → exp(z)·Γ(2,z) = 1 + z
#   Γ(3, z) = (2 + 2z + z²) exp(-z)        → exp(z)·Γ(3,z) = 2 + 2z + z²
#
# These exercise the lower-series, continued-fraction, and asymptotic branches
# against references that involve no incomplete-gamma code at all.
######################################################

@testset "exp_times_upper_gamma primary path (moderate z, integer s)" begin
    # s = 1: result is identically 1 for all z ≥ 0.
    for z in (0.5, 1.0, 2.0, 5.0, 10.0)
        @test exp_times_upper_gamma(1.0, z) ≈ 1.0 rtol=1e-13 atol=0.0
    end

    # s = 2: result is 1 + z.
    for z in (0.5, 1.0, 2.0, 5.0, 10.0)
        @test exp_times_upper_gamma(2.0, z) ≈ 1.0 + z rtol=1e-13 atol=0.0
    end

    # s = 3: result is 2 + 2z + z².
    for z in (0.5, 1.0, 2.0, 5.0, 10.0)
        ref = 2.0 + 2.0*z + z^2
        @test exp_times_upper_gamma(3.0, z) ≈ ref rtol=1e-13 atol=0.0
    end
end


######################################################
# exp_times_upper_gamma — z = 0 case and large-z asymptotic branch
#
# z = 0 special branch:
#   E(s, 0) = exp(0) · Γ(s, 0) = Γ(s)  (upper incomplete gamma at 0 = full Γ)
#
# Large-z branch (z ≥ asymp_cutoff = 128): the asymptotic expansion is used;
# for integer s it terminates exactly, giving the closed forms above.
######################################################

@testset "exp_times_upper_gamma z=0 returns Γ(s)" begin
    for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        @test exp_times_upper_gamma(s, 0.0) ≈ gamma(s) rtol=1e-13 atol=0.0
    end
end

@testset "exp_times_upper_gamma large-z asymptotic branch (integer s, closed forms)" begin
    z = 800.0

    # s = 1: result = 1
    @test exp_times_upper_gamma(1.0, z) ≈ 1.0 rtol=1e-13 atol=0.0

    # s = 2: result = 1 + z
    @test exp_times_upper_gamma(2.0, z) ≈ 1.0 + z rtol=1e-13 atol=0.0

    # s = 3: result = 2 + 2z + z²
    @test exp_times_upper_gamma(3.0, z) ≈ 2.0 + 2.0*z + z^2 rtol=1e-13 atol=0.0
end


######################################################
# exp_times_upper_gamma — BigFloat reference sweep over the paper range
#
# The perturbation solution uses s = 2/β and 3/β with β = α+2, α ∈ (0,1),
# i.e. s ∈ (2/3, 3/2). z = (a/β)(x/L_ε)^β spans 0 up to ~1e20 at small ε.
# The z grid crosses all three algorithm branches (lower series z < s+1,
# continued fraction s+1 ≤ z < 128, asymptotic z ≥ 128) including points
# straddling both branch boundaries — the BigFloat reference arbitrates.
#
# α is given as an exact rational p/q so that α, β = α+2 and s = 2/β, 3/β are
# all built natively in the type under test. Computing s in Float64 and widening
# afterwards would only ever probe Float64-representable arguments, so the extra
# Double64 precision would go untested.
######################################################

# α = p/q for the swept exponents.
const EG_ALPHA_PQ = ((1, 10), (3, 10), (1, 2), (7, 10), (9, 10))

"""
Return `(2/β, 3/β)` for `α = p/q` and `β = α + 2`, constructed natively in `T`.
"""
function eg_s_pair(::Type{T}, p::Integer, q::Integer) where {T<:AbstractFloat}
    β = T(p) / T(q) + T(2)
    return (T(2) / β, T(3) / β)
end

"""
z grid for exponent `s`, built natively in `T`, crossing all three branches:
lower series (z < s+1), continued fraction (s+1 ≤ z < 128), and asymptotic
(z ≥ 128), with points straddling both boundaries and reaching the z ≈ 1e20
encountered at the smallest ε.
"""
function eg_z_grid(::Type{T}, s::T) where {T<:AbstractFloat}
    tiny = T(10)^-8
    return T[
        zero(T), tiny, T(1) / T(10),                     # lower series
        s + one(T) - tiny, s + one(T) + tiny,            # straddles z = s+1
        T(5), T(1279) / T(10), T(1281) / T(10),          # straddles z = 128
        T(10)^3, T(10)^4, T(10)^6, T(10)^8,              # asymptotic
        T(10)^12, T(10)^16, T(10)^20,
    ]
end

@testset "exp_times_upper_gamma vs BigFloat reference, Float64 (EG-1)" begin
    for (p, q) in EG_ALPHA_PQ
        for s in eg_s_pair(Float64, p, q)
            for z in eg_z_grid(Float64, s)
                ref = E_bigfloat_ref(s, z)
                @test relerr(exp_times_upper_gamma(s, z), ref) ≤ 1e-12
            end
        end
    end
end

@testset "exp_times_upper_gamma vs BigFloat reference, Double64 (EG-2)" begin
    for (p, q) in EG_ALPHA_PQ
        for s in eg_s_pair(Double64, p, q)
            for z in eg_z_grid(Double64, s)
                ref = E_bigfloat_ref(s, z)
                @test relerr(exp_times_upper_gamma(s, z), ref) ≤ 1e-27
            end
        end
    end
end


######################################################
# exp_times_upper_gamma — branch boundaries at ULP resolution
#
# The two dispatch boundaries in `exp_times_upper_gamma` are
#
#   z < s + 1          → lower series,  else continued fraction / asymptotic
#   z ≥ asymp_cutoff   → asymptotic tried first (cutoff = 128)
#
# Each is probed at `prevfloat`, the exact cutoff, and `nextfloat`, so that the
# last value handled by one branch and the first handled by the next are both
# checked. `z = s+1` exactly and `z = 128` exactly take the *upper* branch in
# each case, since both comparisons are `<` / `≥`.
#
# This is the tightest spot in the whole range: the continued fraction converges
# most slowly just above z = s+1, where the observed error reaches ≈ 2·64eps(T)
# (the routine's own convergence tolerance). The assertions below are set an
# order of magnitude above that, not at eps(T).
######################################################

@testset "exp_times_upper_gamma branch boundaries, Float64 (EG-4)" begin
    for (p, q) in EG_ALPHA_PQ
        for s in eg_s_pair(Float64, p, q)
            cutoffs = (s + 1.0, 128.0)
            for zc in cutoffs
                for z in (prevfloat(zc), zc, nextfloat(zc))
                    ref = E_bigfloat_ref(s, z)
                    @test relerr(exp_times_upper_gamma(s, z), ref) ≤ 1e-12
                end
            end
        end
    end
end

@testset "exp_times_upper_gamma branch boundaries, Double64 (EG-5)" begin
    for (p, q) in EG_ALPHA_PQ
        for s in eg_s_pair(Double64, p, q)
            cutoffs = (s + one(Double64), Double64(128))
            for zc in cutoffs
                for z in (prevfloat(zc), zc, nextfloat(zc))
                    ref = E_bigfloat_ref(s, z)
                    @test relerr(exp_times_upper_gamma(s, z), ref) ≤ 1e-27
                end
            end
        end
    end
end


######################################################
# exp_times_upper_gamma — the two upper branches must agree with each other
#
# Above z = s+1 the value can be obtained two ways, and `asymp_cutoff` selects
# between them: `Inf` forces the continued fraction, `z` forces the asymptotic
# expansion. Where both are valid they must agree, which checks the asymptotic
# branch against the continued fraction with no reference implementation of the
# incomplete gamma involved at all — independent of MPFR and of `loggamma`.
######################################################

@testset "exp_times_upper_gamma continued fraction vs asymptotic agreement (EG-6)" begin
    for (T, rtol) in ((Float64, 1e-13), (Double64, 1e-28))
        for (p, q) in EG_ALPHA_PQ
            for s in eg_s_pair(T, p, q)
                for z in (T(128), T(10)^3, T(10)^6, T(10)^8, T(10)^12, T(10)^20)
                    cf   = exp_times_upper_gamma(s, z; asymp_cutoff = T(Inf))
                    asym = exp_times_upper_gamma(s, z; asymp_cutoff = z)
                    @test abs(cf - asym) ≤ rtol * abs(cf)
                end
            end
        end
    end
end


######################################################
# exp_times_upper_gamma — noninteger s on the asymptotic branch
#
# The integer-s closed forms above terminate the asymptotic series exactly, so
# they do not exercise its truncation-and-tolerance logic. These spot checks pin
# the asymptotic branch for noninteger s in the paper range, forcing that branch
# explicitly via `asymp_cutoff` so the test cannot silently fall through to the
# continued fraction if the cutoff is ever retuned.
######################################################

@testset "exp_times_upper_gamma asymptotic branch, noninteger s (EG-7)" begin
    for (T, tol) in ((Float64, 1e-12), (Double64, 1e-27))
        for s in (T(2) / (T(1) / T(10) + T(2)), T(3) / (T(9) / T(10) + T(2)))
            for z in (T(10)^3, T(10)^6, T(10)^8, T(10)^16, T(10)^20)
                ref = E_bigfloat_ref(s, z)
                @test relerr(exp_times_upper_gamma(s, z; asymp_cutoff = z), ref) ≤ tol
            end
        end
    end
end

@testset "construct_perturbation_solution end-to-end vs BigFloat, Double64 (EG-3)" begin
    α = Double64(0.5); ε = Double64(1e-9); rho_half = Double64(1.2)
    y = construct_perturbation_solution(α, ε, rho_half; order=:second)

    αb = BigFloat(α)
    function y_ref(x)
        setprecision(BigFloat, 512) do
            βb  = αb + 2
            ab  = exp2(αb)
            Lb  = (BigFloat(ε)^2 / 6)^(1 / βb)
            zb  = (ab / βb) * (BigFloat(x) / Lb)^βb
            K2b = (BigFloat(rho_half) / βb^2) * (βb / ab)^(2 / βb)
            K3b = (BigFloat(rho_half) * αb * (1 + αb) / βb^4) * (βb / ab)^(3 / βb)
            U   = K2b * exp(zb + loggamma(2 / βb, zb)) +
                  Lb * K3b * exp(zb + loggamma(3 / βb, zb))
            return U * Lb^(-αb)
        end
    end

    for x in 10.0 .^ range(-12, -0.31; length=13)
        @test relerr(y(Double64(x)), y_ref(x)) ≤ 1e-27
    end
end

######################################################
# construct_perturbation_solution — noisy → noise-free limit
#
# For x well outside the boundary layer (x >> L = (αε²/(2^α 6)^(1/(2+α))), the noisy
# constructor reduces to the noise-free power law. With epsilon = 1e-6, the
# boundary layer scale L is O(1e-3) or smaller for all α ∈ (0,1), so at
# x = 0.1..0.4 we are deep in the outer region and the two closures should
# agree to better than 1% relative error.
######################################################

@testset "construct_perturbation_solution: noisy converges to noise-free for small epsilon (PS-3)" begin
    T        = Float64
    rho_half = T(1.0)
    epsilon  = T(1e-6)
    xs       = T[0.1, 0.2, 0.3, 0.4]

    for alpha in T[0.3, 0.5, 0.7]
        noisy     = construct_perturbation_solution(alpha, epsilon, rho_half)
        noisefree = construct_perturbation_solution(alpha, rho_half)

        for x in xs
            ny  = noisy(x)
            nfy = noisefree(x)

            # Both closures must return positive finite values.
            @test isfinite(ny)  && ny  > zero(T)
            @test isfinite(nfy) && nfy > zero(T)

            # In the outer region the two formulas agree to better than 1%.
            @test ny ≈ nfy rtol=1e-2
        end
    end
end



