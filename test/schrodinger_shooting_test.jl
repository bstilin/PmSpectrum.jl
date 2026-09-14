# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

# -------------------------------------
# Tests for SchrodingerShooting.jl
#
# Matched shooting for -y'' + q_α y = λ y on (0, L), by parity.
#
#   SH-1   phase-vector conventions: initial data in, (y, y') back out
#   SH-2   turning_point inverts q_α
#   SH-3   raw and normalized Wronskians have the same root
#   SH-4   residual is small at the returned eigenvalue
#   SH-5   FD → shooting agreement, eigenvalues
#   SH-6   FD → shooting agreement, eigenfunctions
#   SH-7   xmatch insensitivity
#   SH-8   eigenfunction internal consistency
#   SH-9   a non-bracketing interval errors clearly
#   SH-11  sweep helpers and the bracket-partition assumption
#   SH-12  known answer: free particle
#   SH-13  known answer: harmonic oscillator
# -------------------------------------


using PmSpectrum.SchrodingerFD: potential, solve_fd
using PmSpectrum.SchrodingerShooting: left_initial_data, right_initial_data, phase_state,
    shooting_data, matching_residual, refine_eigenvalue, shooting_eigenfunction,
    turning_point, bracket_from_fd, brackets_from_fd, refine_spectrum, merge_parity


######################################################
# Helpers and conventions
######################################################

@testset "phase-vector conventions: initial data in, (y, y') back out (SH-1)" begin
    # The three functions that define the phase-vector convention, tested as one
    # round trip: `left_initial_data`/`right_initial_data` impose (y, y') at the
    # two ends, and `phase_state` reads it back in the same order.

    α, L, λ = 0.5, 8.0, 1.51

    for parity in (:even, :odd)
        y0, dy0 = left_initial_data(parity, Float64)
        d = shooting_data(α, λ, L, 1.0; parity=parity)

        # at x = 0 the left solution must reproduce the initial data, in order
        y, dy = phase_state(d.left_solution, 0.0)
        @test y  ≈ y0  atol=1e-12
        @test dy ≈ dy0 atol=1e-12

        # ...and the raw SciML state is the reverse of it
        raw = d.left_solution(0.0)
        @test raw[1] ≈ dy0 atol=1e-12
        @test raw[2] ≈ y0  atol=1e-12
    end

    @test left_initial_data(:even, Float64) == (1.0, 0.0)   # y'(0) = 0
    @test left_initial_data(:odd,  Float64) == (0.0, 1.0)   # y(0)  = 0
    @test right_initial_data(0.5, 1.5, 8.0) == (0.0, 1.0)   # y(L)  = 0
    @test_throws ArgumentError left_initial_data(:sideways, Float64)
    @test_throws ArgumentError right_initial_data(0.5, 1.5, 8.0; right_boundary=:wkb)
end


@testset "turning_point inverts q_α (SH-2)" begin
    for α in (0.15, 0.5, 0.85), λ in (0.5, 1.5, 10.0, 40.0)
        xt = turning_point(α, λ)
        @test xt > 0
        @test potential(xt, α) ≈ λ rtol=1e-12
    end
    @test_throws ArgumentError turning_point(0.5, -1.0)
end


######################################################
# The matching residual
######################################################

@testset "raw and normalized Wronskians share the root (SH-3)" begin
    α, L = 0.5, 8.0
    fd = solve_fd(α, L, 4000; parity=:even, nev=3)
    xm = turning_point(α, fd.eigenvalues[1])
    br = bracket_from_fd(fd.eigenvalues, 1)

    a = refine_eigenvalue(α, br, L, xm; parity=:even, normalized=true)
    b = refine_eigenvalue(α, br, L, xm; parity=:even, normalized=false)
    @test a.eigenvalue ≈ b.eigenvalue rtol=1e-12

    # W_norm = W_raw / (‖Y_L‖‖Y_R‖), a strictly positive rescaling — which is
    # *why* the two share every zero and every sign. Checked away from the root:
    # near it W_raw is a difference of two enormous nearly-equal products and
    # loses its relative precision to cancellation.
    for λ in (br[1], 1.2, 2.0, br[2])
        d = shooting_data(α, λ, L, xm; parity=:even)
        nL, nR = hypot(d.left_state...), hypot(d.right_state...)
        @test d.raw_wronskian / (nL * nR) ≈ d.normalized_wronskian rtol=1e-12
        @test signbit(d.raw_wronskian) == signbit(d.normalized_wronskian)
    end

    # the normalized residual stays O(1); the raw one spans many orders
    @test abs(a.normalized_wronskian) ≤ 1 + 1e-8
end


@testset "residual is small at the returned eigenvalue (SH-4)" begin
    α, L = 0.5, 8.0
    for parity in (:even, :odd)
        fd = solve_fd(α, L, 4000; parity=parity, nev=3)
        for j in 1:2
            xm = turning_point(α, fd.eigenvalues[j])
            sh = refine_eigenvalue(α, bracket_from_fd(fd.eigenvalues, j), L, xm; parity=parity)
            @test abs(sh.residual) < 1e-10
            @test sh.convergence_flag === :x_converged
            @test sh.bracket[1] < sh.eigenvalue < sh.bracket[2]
            @test signbit(sh.bracket_residuals[1]) != signbit(sh.bracket_residuals[2])
        end
    end
end


######################################################
# Agreement with finite differences
######################################################

@testset "FD → shooting, eigenvalues (SH-5)" begin
    # The two methods must solve the SAME finite-L Dirichlet problem, so the FD
    # eigenvalue should approach the shooting value as N grows.
    L = 8.0
    for α in (0.15, 0.5, 0.85), parity in (:even, :odd)
        fd0 = solve_fd(α, L, 2000; parity=parity, nev=3)
        xm  = turning_point(α, fd0.eigenvalues[1])
        sh  = refine_eigenvalue(α, bracket_from_fd(fd0.eigenvalues, 1), L, xm; parity=parity)

        errs = [abs(solve_fd(α, L, N; parity=parity, nev=1).eigenvalues[1] - sh.eigenvalue)
                for N in (1000, 2000, 4000, 8000)]
        @test issorted(errs; rev=true)     # monotone convergence toward shooting
        @test errs[end] < errs[1] / 3
    end
end


@testset "FD → shooting, eigenfunctions (SH-6)" begin
    # Eigenvalues can be nearly right while the eigenfunction is wrong (bad
    # parity, bad splice constant, reversed phase_state), so compare functions.
    # Both solvers already normalise to unit L² on [0, L], so no rescaling is
    # needed — only the arbitrary sign has to be aligned.
    #
    # dtmax is set because the returned callable goes through DPRKN12's dense
    # output, which is far lower order than the integrator; with the default
    # dtmax the shooting side, not FD, would be the limiting error.
    L = 8.0
    for α in (0.5, 0.85), parity in (:even, :odd)
        fd0 = solve_fd(α, L, 2000; parity=parity, nev=2)
        xm  = turning_point(α, fd0.eigenvalues[1])
        sh  = refine_eigenvalue(α, bracket_from_fd(fd0.eigenvalues, 1), L, xm; parity=parity)
        ef  = shooting_eigenfunction(α, sh.eigenvalue, L, xm; parity=parity, dtmax=0.01)

        errs = map((2000, 8000)) do N
            fd = solve_fd(α, L, N; parity=parity, nev=1)
            ys = [ef.eigenfunction(x) for x in fd.x]
            u  = copy(fd.eigenfunctions[:, 1])
            u .*= sign(sum(u .* ys))           # align sign by correlation, not one sample
            maximum(abs, ys .- u)
        end

        @test errs[2] < errs[1]      # refining FD moves it toward shooting
        @test errs[2] < 1e-4
    end
end


@testset "xmatch insensitivity (SH-7)" begin
    α, L = 0.5, 8.0
    fd = solve_fd(α, L, 4000; parity=:even, nev=3)
    xt = turning_point(α, fd.eigenvalues[1])
    br = bracket_from_fd(fd.eigenvalues, 1)

    λs = [refine_eigenvalue(α, br, L, f * xt; parity=:even).eigenvalue
          for f in (0.7, 1.0, 1.3)]
    @test maximum(λs) - minimum(λs) < 1e-11
end


######################################################
# Eigenfunction reconstruction
######################################################

@testset "eigenfunction internal consistency (SH-8)" begin
    α, L = 0.5, 8.0
    for parity in (:even, :odd)
        fd = solve_fd(α, L, 4000; parity=parity, nev=2)
        xm = turning_point(α, fd.eigenvalues[1])
        sh = refine_eigenvalue(α, bracket_from_fd(fd.eigenvalues, 1), L, xm; parity=parity)
        ef = shooting_eigenfunction(α, sh.eigenvalue, L, xm; parity=parity, dtmax=0.01)

        # the two phase vectors are proportional at the matching point
        @test all(abs.(ef.left_state .- ef.match_scale .* ef.right_state) .< 1e-10)

        # unit L² norm on [0, L]
        n2, _ = quadgk(x -> ef.eigenfunction(x)^2, 0.0, xm, L; rtol=1e-12)
        @test n2 ≈ 1.0 rtol=1e-8

    end
end


@testset "a non-bracketing interval errors clearly (SH-9)" begin
    α, L = 0.5, 8.0
    xm = turning_point(α, 1.51)

    # (1.0, 1.2) is chosen from the known spectrum: the even eigenvalues at
    # α = 0.5, L = 8 are 1.5100, 5.9605, 11.0420, …, so this interval lies
    # entirely BELOW the ground state and encloses no root. F is then the same
    # sign at both ends (+0.610 and +0.378) and refine_eigenvalue must reject it
    # rather than hand the bracket to the root finder.
    # The message is the feature: it reports both endpoint residuals so the
    # caller can see how far off the bracket is. `@test_throws` matches a string
    # against the rendered message, so the type and the text are both pinned.
    @test_throws ArgumentError    refine_eigenvalue(α, (1.0, 1.2), L, xm; parity=:even)
    @test_throws "does not straddle" refine_eigenvalue(α, (1.0, 1.2), L, xm; parity=:even)
    @test_throws "F(lo)"          refine_eigenvalue(α, (1.0, 1.2), L, xm; parity=:even)
    @test_throws "F(hi)"          refine_eigenvalue(α, (1.0, 1.2), L, xm; parity=:even)

    # A reversed interval is a different error path — caught by the lo < hi
    # check before F is evaluated at all, so it needs no knowledge of the
    # spectrum.
    @test_throws "lo < hi"        refine_eigenvalue(α, (2.0, 1.0), L, xm; parity=:even)
end


######################################################
# Sweep helpers
######################################################

@testset "refine_spectrum and merge_parity (SH-11)" begin
    L, nev = 8.0, 6
    for α in (0.15, 0.85)
        sectors = map((:even, :odd)) do parity
            fd = solve_fd(α, L, 4000; parity=parity, nev=nev)
            refine_spectrum(α, brackets_from_fd(fd.eigenvalues), L,
                            turning_point.(α, fd.eigenvalues); parity=parity)
        end

        for s in sectors
            @test issorted(s.eigenvalues)
            @test length(s.results) == nev
            @test all(s.brackets[j][1] ≤ s.eigenvalues[j] ≤ s.brackets[j][2] for j in 1:nev)
        end

        # a single-mode call reproduces the swept result exactly
        one = refine_eigenvalue(α, sectors[1].brackets[2], L, sectors[1].xmatches[2];
                                parity=:even)
        @test one.eigenvalue == sectors[1].eigenvalues[2]

        m = merge_parity(sectors...)
        @test issorted(m.eigenvalues)
        @test length(m.eigenvalues) == 2nev
        @test m.interleaved                      # even, odd, even, … per nodal theory
        @test m.parities[1] === :even
        @test m.results[1].eigenvalue == m.eigenvalues[1]
        # sector_index points back into the right sector
        @test m.eigenvalues[1] == sectors[1].eigenvalues[m.sector_index[1]]
        @test m.eigenvalues[2] == sectors[2].eigenvalues[m.sector_index[2]]
    end

    # The brackets are the assumption `refine_spectrum` rests on, here we check the assumptions
    α = 0.5
    fd = solve_fd(α, L, 4000; parity=:even, nev=nev)
    brs = brackets_from_fd(fd.eigenvalues)

    @test all(b[1] < b[2] for b in brs)                                  # non-empty
    @test issorted([b[1] for b in brs]) && issorted([b[2] for b in brs]) # increasing
    @test all(brs[j][2] === brs[j+1][1] for j in 1:length(brs)-1)        # abut, bitwise
    @test all(brs[j][1] < fd.eigenvalues[j] < brs[j][2] for j in eachindex(fd.eigenvalues))

    # a sign change needs an ODD number of roots, so a bracket holding zero or two is
    # rejected outright.
    xm = turning_point(α, fd.eigenvalues[1])
    @test_throws ArgumentError refine_eigenvalue(α, (2.6, 4.8), L, xm; parity=:even)  # 0 roots
    @test_throws ArgumentError refine_eigenvalue(α, (0.0, 7.0), L, xm; parity=:even)  # 2 roots
    @test refine_eigenvalue(α, (0.0, 3.0), L, xm; parity=:even).eigenvalue ≈
          fd.eigenvalues[1] rtol=1e-5                                                 # 1 root
end


######################################################
# Known-answer checks, via the `q` hook
######################################################

# Brackets are hand-chosen from the known spectrum rather than taken from
# `solve_fd`, and `xmatch` is likewise explicit — for the free particle there is
# no turning point to compute, since q ≡ 0 never reaches λ.
_free_sh(x, a) = zero(x)
_sq_sh(x, a)   = x^2

@testset "known answer: free particle (SH-12)" begin
    # -y'' = λy.  even (y'(0)=0, y(L)=0): λ_k = ((k-½)π/L)²
    #             odd  (y(0)=0,  y(L)=0): λ_k = (kπ/L)²
    for L in (1.0, 3.0)
        for (parity, exact) in ((:even, [((k - 0.5) * π / L)^2 for k in 1:3]),
                                (:odd,  [(k * π / L)^2         for k in 1:3]))
            for ex in exact
                sh = refine_eigenvalue(0.5, (0.6ex, 1.4ex), L, L/2;
                                       parity=parity, q=_free_sh)
                @test sh.eigenvalue ≈ ex rtol=1e-10
                @test abs(sh.residual) < 1e-10
            end
        end
    end
end


@testset "known answer: harmonic oscillator (SH-13)" begin
    # -y'' + x²y = λy on the line ⇒ λ = 2k+1; the even sector picks out
    # 1, 5, 9, … and the odd sector 3, 7, 11, …
    L = 10.0
    for (parity, exact) in ((:even, [1.0, 5.0, 9.0]), (:odd, [3.0, 7.0, 11.0]))
        for ex in exact
            xm = turning_point(0.5, ex; q=_sq_sh)          # q = x² ⇒ x_t = √λ
            @test xm ≈ sqrt(ex) rtol=1e-10

            sh = refine_eigenvalue(0.5, (ex - 1.5, ex + 1.5), L, xm; parity=parity, q=_sq_sh)
            @test sh.eigenvalue ≈ ex rtol=1e-10
            @test abs(sh.residual) < 1e-10
        end
    end

    # and the eigenfunction reconstruction works on the swapped potential too
    ef = shooting_eigenfunction(0.5, 1.0, L, 1.0; parity=:even, q=_sq_sh, dtmax=0.01)
    n2, _ = quadgk(x -> ef.eigenfunction(x)^2, 0.0, 1.0, L; rtol=1e-12)
    @test n2 ≈ 1.0 rtol=1e-8
    # The oscillator ground state is exp(-x²/2) up to normalisation.
    y0 = ef.eigenfunction(0.0)
    for x in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
        @test ef.eigenfunction(x) / y0 ≈ exp(-x^2 / 2) rtol=1e-10
    end
end
