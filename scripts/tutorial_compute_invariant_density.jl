# =============================================================================
#   Standalone, step-by-step computation of the invariant density of the noisy
#   symmetric Pomeau–Manneville map using the cubic B-spline Galerkin method.
#
#   Also computes the Lyapunov exponent, correlation decay, compares the real
#   eigenvalues to the Schrödinger prediction, and estimates the reduced
#   resolvent norm.
#
#   Intended as an interactive, transparent, and tinkerable complement to
#   run_single_experiment.
# =============================================================================

# ENV["MPLBACKEND"] = "tkagg"  # for interactive plotting on some platforms

using PyPlot
using PmSpectrum
using BSplineKit
using DoubleFloats
using LinearAlgebra

close("all")


# ============================================================
# PARAMETERS
# ============================================================

T                = Float64   # Supported types: Float64, Double64
ALPHA            = T(0.85)    # PM map parameter α in (0,1)
EPSILON          = T(1e-9)   # Noise half-width 0 < ε < 1/2
NUM_BREAK_POINTS = 100        # Approximately half the desired number of B-splines
NUM_QUAD_POINTS  = 64         # Quadrature points per piece of the composite
                              # Gaussian quadrature rule for Galerkin matrices


# ============================================================
# BASIS
# ============================================================

basis, bdry_layer_width = Bases.build_pm_basis(
    ALPHA,
    EPSILON,
    NUM_BREAK_POINTS,
    T,
)


# ============================================================
# INVERSE BRANCHES
# ============================================================

left_inv  = Utils.construct_left_branch_inverse(ALPHA)
right_inv = x -> one(T) - left_inv(one(T) - x)
right_fwd = x -> one(T) - Utils.symmetric_pm(one(T) - x, ALPHA)

b1 = Bases.Branch(
    x -> Utils.symmetric_pm(x, ALPHA),
    left_inv,
    (zero(T), T(0.5)),
    (zero(T), one(T)),
)

b2 = Bases.Branch(
    right_fwd,
    right_inv,
    (T(0.5), one(T)),
    (zero(T), one(T)),
)


# ============================================================
# GALERKIN MATRICES
# ============================================================

spls = Bases.build_single_splines(basis)
M = Bases.mass_matrix(spls)

endpoint_refine = Bases.EndpointRefinement(ALPHA)

G = Bases.transfer_matrix(
    spls,
    spls,
    EPSILON,
    (b1, b2);
    n_quad=NUM_QUAD_POINTS,
    atol=eps(T),
    refine=endpoint_refine,
)


# ============================================================
# EIGENSOLVE
# ============================================================

λ_raw, V_raw = Utils.scaled_nonsymmetric_eigen(G, M)

p = sortperm(abs.(λ_raw); rev=true)

λ_sorted = λ_raw[p]
V_sorted = V_raw[:, p]

println("ε = ", EPSILON, "   α = ", ALPHA, "   L_ε = ", bdry_layer_width)
println("Leading eigenvalue  λ₁ = ", λ_sorted[1])
println("Second eigenvalue   λ₂ = ", λ_sorted[2])
println("Modulus spectral gap 1 - |λ₂| = ", 1 - abs(λ_sorted[2]))


# ============================================================
# NUMERICALLY COMPUTED INVARIANT DENSITY
# ============================================================
# The invariant density is represented by the leading eigenvector and
# normalized to have unit mass.

c_raw = real.(V_sorted[:, 1])

mass = sum(
    c_raw[i] * spls[i].mass
    for i in eachindex(c_raw)
)

c = c_raw / mass

pdf = BSplineKit.Splines.Spline(basis, c)


# ============================================================
# EVALUATION GRID
# ============================================================

xs = Utils.symmetric_logspace_grid(
    2001;
    decades=-17,
    include_mid=true,
    include_ends=false,
    T=T,
)

# Numerical invariant density evaluated on the grid.
pdf_grid = pdf.(xs)

# Formal asymptotic boundary-layer prediction.
perturb = Utils.construct_perturbation_solution(
    ALPHA,
    EPSILON,
    pdf(T(0.5));
    order=:second,
)

perturb_grid = perturb.(xs)

xstar = bdry_layer_width


# ============================================================
# PLOT 1 — Invariant density vs. formal boundary-layer prediction
# ============================================================

fig, ax = subplots(figsize=(7, 5))

ax.plot(
    Float64.(xs),
    Float64.(pdf_grid);
    color="steelblue",
    linewidth=1.5,
    label=raw"numerical $\rho(x)$",
)

ax.plot(
    Float64.(xs),
    Float64.(perturb_grid);
    color="firebrick",
    linestyle=(0, (4, 3)),
    linewidth=1.3,
    label=raw"boundary layer prediction $\rho(x)$",
)

ax.axvline(
    Float64(xstar);
    color="dimgray",
    linewidth=2.0,
    linestyle="--",
    alpha=0.85,
    label="boundary layer scale",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("x")
ax.set_ylabel(raw"$\rho(x)$")
ax.set_title("Invariant Density  (α=$(ALPHA), ε=$(EPSILON))")
ax.legend()
ax.grid(true, alpha=0.3)

fig.tight_layout()


# ============================================================
# PLOT 2 — Relative error in the boundary-layer prediction
# ============================================================

rel_err = abs.(pdf_grid .- perturb_grid) ./ abs.(pdf_grid)

fig2, ax2 = subplots(figsize=(7, 5))

ax2.plot(
    Float64.(xs),
    Float64.(rel_err);
    color="steelblue",
    linewidth=1.5,
)

ax2.axvline(
    Float64(xstar);
    color="dimgray",
    linewidth=2.0,
    linestyle="--",
    alpha=0.85,
    label="boundary layer scale",
)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("x")
ax2.set_ylabel(raw"$|\rho_\mathrm{num} - \rho_\mathrm{pred}| \,/\, \rho_\mathrm{num}$")
ax2.set_title("Relative Error: Numerical vs. Predicted  (α=$(ALPHA), ε=$(EPSILON))")
ax2.legend()
ax2.grid(true, alpha=0.3)

fig2.tight_layout()


# ============================================================
# PLOT 3 — Eigenvalue spectrum
# ============================================================

λ_re = Float64.(real.(λ_sorted))
λ_im = Float64.(imag.(λ_sorted))
θ = range(0, 2π; length=400)

fig3, ax3 = subplots(figsize=(6, 6))

ax3.scatter(
    λ_re,
    λ_im;
    color="steelblue",
    s=20,
    zorder=3,
)

ax3.plot(
    cos.(θ),
    sin.(θ);
    color="black",
    linestyle="--",
    linewidth=0.8,
    label="unit circle",
)

ax3.set_aspect("equal")
ax3.set_xlabel(raw"Re($\lambda$)")
ax3.set_ylabel(raw"Im($\lambda$)")
ax3.set_title("Eigenvalue Spectrum  (α=$(ALPHA), ε=$(EPSILON))")
ax3.grid(true, alpha=0.3)
ax3.legend()

fig3.tight_layout()


# ============================================================
# LYAPUNOV EXPONENT
# ============================================================
# The Lyapunov exponent is
#
#     λ_Lyap = ∫₀¹ log T_α'(x) ρ_ε(x) dx,
#
# i.e. the ρ_ε-average of the local expansion rate.
#
# With d = min(x, 1-x),
#
#     T_α'(x) = 1 + (1+α) 2^α d^α ≥ 1.

dq = density_quadrature(
    pdf,
    spls[1].knots,
    EPSILON;
    n_quad=NUM_QUAD_POINTS,
    atol=eps(T),
    refine=endpoint_refine,
)

lyapunov = integrate_against_density(
    x -> Utils.log_symmetric_pm_derivative(x, ALPHA),
    dq,
)

println("\n--- Lyapunov exponent ---")
println("λ_Lyap = ∫ log T'_α ρ_ε = ", lyapunov)
println("∫ρ_ε (must be 1)        = ", total_mass(dq))


# ============================================================
# PLOT 4 — Lyapunov integrands
# ============================================================
# The integrand log T'(x) ρ(x), shown on both log-linear and
# ordinary linear x-scales.

lyap_density = Utils.log_symmetric_pm_derivative.(xs, ALPHA) .* pdf_grid

fig4, ax4 = subplots(1, 2, figsize=(10, 5))

fig4.suptitle("Lyapunov Integrands")

ax4[1].plot(
    Float64.(xs),
    Float64.(lyap_density);
    color="steelblue",
    linewidth=1.5,
)

ax4[1].axvline(
    Float64(xstar);
    color="dimgray",
    linewidth=2.0,
    linestyle="--",
    alpha=0.85,
    label="boundary layer scale",
)

ax4[1].set_xscale("log")
ax4[1].set_xlabel("x")
ax4[1].set_ylabel(raw"$\log T_\alpha'(x)\,\rho(x)$")
ax4[1].set_title("Log-Linear Scale")
ax4[1].legend()
ax4[1].grid(true, alpha=0.3)

ax4[2].plot(
    Float64.(xs),
    Float64.(lyap_density);
    color="steelblue",
    linewidth=1.5,
)

ax4[2].set_xlabel("x")
ax4[2].set_ylabel(raw"$\log T_\alpha'(x)\,\rho(x)$")
ax4[2].set_title("Linear Scale")
ax4[2].grid(true, alpha=0.3)

fig4.tight_layout()


# ============================================================
# CORRELATION DECAY CURVES
# ============================================================
# For a centred observable
#
#     f₀ = f - ∫ f ρ_ε,
#
# the stationary autocorrelation is
#
#     C(n) = ∫ f₀(Tⁿx) f₀(x) dμ_ε(x).
#
# With an isolated spectral gap, we expect the long-lag correlation to
# decay exponentially:
#
#     |C(n)| ≈ c |λ|ⁿ.
#
# Thus
#
#     log|C(n)| ≈ log c + n log|λ|,
#
# so fitting an exponential is equivalent to a least-squares linear fit
# of log|C(n)| against n.
#
# The map is symmetric under x ↦ 1-x. Numerical eigenvectors and the
# Schrödinger prediction suggest that the leading nontrivial eigenfunction
# (λ₂) is even, while the next eigenfunction (λ₃) is odd. We therefore compare
#
#     cos(2πx)  — even under x ↦ 1-x,
#     sin(2πx)  — odd  under x ↦ 1-x.
#
# We expect the correlation of even observables to decay asymptotically like |λ₂|ⁿ and
# the correlation of odd observables like |λ₃|ⁿ. Since |λ₂| > |λ₃|, the even correlation
# should decay more slowly than the odd correlation.
#
# We fit the last third of each correlation curve and compare the fitted
# per-lag decay factors with |λ₂| and |λ₃|, respectively.

even_observable(x) = cos(2 * T(π) * x)
odd_observable(x)  = sin(2 * T(π) * x)

MAX_LAG = 10_000

EVEN_MODE_INDEX = 2
ODD_MODE_INDEX  = 3

λ_even = λ_sorted[EVEN_MODE_INDEX]
λ_odd  = λ_sorted[ODD_MODE_INDEX]


# ============================================================
# EVEN CORRELATION
# ============================================================

cc_even = correlation_curve(
    even_observable,
    spls,
    M,
    G,
    c,
    EPSILON,
    MAX_LAG;
    n_quad=NUM_QUAD_POINTS,
    atol=eps(T),
    refine=endpoint_refine,
)

fit_even = estimate_decay_rate(cc_even)


# ============================================================
# ODD CORRELATION
# ============================================================

cc_odd = correlation_curve(
    odd_observable,
    spls,
    M,
    G,
    c,
    EPSILON,
    MAX_LAG;
    n_quad=NUM_QUAD_POINTS,
    atol=eps(T),
    refine=endpoint_refine,
)

fit_odd = estimate_decay_rate(cc_odd)

fit_start = MAX_LAG - cld(MAX_LAG + 1, 3) + 1


# ============================================================
# CORRELATION DIAGNOSTICS
# ============================================================

even_rel_err = abs(abs(λ_even) - fit_even.rate) / abs(fit_even.rate)
odd_rel_err  = abs(abs(λ_odd)  - fit_odd.rate)  / abs(fit_odd.rate)

println("\n--- correlation decay ---")
println("fit interval = ", fit_start, ":", MAX_LAG)

println("\nEven observable:  cos(2πx)")
println("fitted rate        = ", fit_even.rate)
println("predicted |λ₂|     = ", abs(λ_even))
println("relative error     = ", even_rel_err)

println("\nOdd observable:   sin(2πx)")
println("fitted rate        = ", fit_odd.rate)
println("predicted |λ₃|     = ", abs(λ_odd))
println("relative error     = ", odd_rel_err)


# ============================================================
# PLOT 5 — Even and odd correlation decay
# ============================================================

lags = Float64.(collect(cc_even.lags))

C_even = Float64.(abs.(cc_even.C))
C_odd  = Float64.(abs.(cc_odd.C))

C_fit_even = Float64(fit_even.amplitude) .* Float64(fit_even.rate) .^ lags
C_fit_odd  = Float64(fit_odd.amplitude)  .* Float64(fit_odd.rate)  .^ lags

fig5, ax5 = subplots(figsize=(7, 5))

ax5.plot(
    lags,
    C_even;
    color="steelblue",
    linewidth=1.5,
    label=raw"even: $|C(n)|$",
)

ax5.plot(
    lags,
    C_fit_even;
    color="steelblue",
    linestyle="--",
    linewidth=1.2,
    label="even fit = $(round(Float64(fit_even.rate); sigdigits=8))ⁿ,  |λ₂| = $(round(Float64(abs(λ_even)); sigdigits=8))",
)

ax5.plot(
    lags,
    C_odd;
    color="firebrick",
    linewidth=1.5,
    label=raw"odd: $|C(n)|$",
)

ax5.plot(
    lags,
    C_fit_odd;
    color="firebrick",
    linestyle="--",
    linewidth=1.2,
    label="odd fit = $(round(Float64(fit_odd.rate); sigdigits=8))ⁿ,  |λ₃| = $(round(Float64(abs(λ_odd)); sigdigits=8))",
)

ax5.set_yscale("log")
ax5.set_xlabel("lag n")
ax5.set_ylabel(raw"$|C(n)|$")
ax5.set_title("Even and Odd Correlation Decay  (α=$(ALPHA), ε=$(EPSILON))")
ax5.legend()
ax5.grid(true, alpha=0.3)

fig5.tight_layout()


# ============================================================
# SCHRÖDINGER PREDICTION OF THE REAL EIGENVALUES
# ============================================================
# After rescaling by the boundary layer, we predict that the real spectrum
# near λ = 1 is governed by the full-line Schrödinger operator
#
#     H = -d²/dy² + q_α(|y|),
#
#     q_α(y) = 2^(α-1)(1+α)y^α + 2^(2α-2)y^(2+2α),
#
# whose eigenvalues
#
#     0 < ω₀ < ω₁ < ⋯
#
# predict nontrivial real transfer-operator eigenvalues of the form
#
#     λ_j^pred(ε) = exp(-ω_j L_ε^α)
#                 = 1 - ω_j L_ε^α + ⋯ .
#
# This prediction concerns the REAL eigenvalues near λ = 1, ordered from
# largest to smallest after the invariant eigenvalue λ = 1 is removed.
# Complex eigenvalues are not included in the numerical comparison.
#
# We report two different relative errors:
#
#   1. Eigenvalue relative error
#
#          |λ_pred - λ_meas| / |λ_meas|,
#
#      which measures the relative accuracy of the predicted eigenvalue itself.
#
#   2. Gap relative error
#
#          |(1 - λ_pred) - (1 - λ_meas)| / |1 - λ_meas|,
#
#      which measures the relative accuracy of the predicted distance from
#      λ = 1.

SCH_L = 8.0   # truncation half-width; eigenfunctions negligible past ~8
SCH_N = 4000  # FD grid points, used only to bracket the shooting problem
N_SEC = 6     # modes per parity sector, so 2*N_SEC predictions total

αf = Float64(ALPHA)

sl = SchrodingerShooting.merge_parity(
    map((:even, :odd)) do parity
        fd = SchrodingerFD.solve_fd(
            αf,
            SCH_L,
            SCH_N;
            parity=parity,
            nev=N_SEC,
        )

        SchrodingerShooting.refine_spectrum(
            αf,
            SchrodingerShooting.brackets_from_fd(fd.eigenvalues),
            SCH_L,
            SchrodingerShooting.turning_point.(αf, fd.eigenvalues);
            parity=parity,
        )
    end...,
)

L_eps = Float64(bdry_layer_width)  # = (ε²/6)^(1/(2+α))

predicted = TransferPrediction.predicted_eigenvalues(
    sl.eigenvalues,
    αf,
    L_eps,
)  # exp(-ω_j L_ε^α)


# ============================================================
# MEASURED REAL EIGENVALUES
# ============================================================
# Keep only real eigenvalues, sort them from largest to smallest, and remove
# the invariant eigenvalue λ = 1.
#
# A complex pair near 1, which can appear for larger ε, is not described by
# this Schrödinger prediction and is therefore filtered out.
#
# Round-off imaginary parts should lie far below √eps(T), while a genuine
# complex pair should lie far above it.

IM_TOL = sqrt(eps(T))

measured = sort!(
    real.(filter(λ -> abs(imag(λ)) <= IM_TOL, λ_sorted));
    rev=true,
)[2:end]

n_cmp = min(length(predicted), length(measured))


# ============================================================
# SCHRÖDINGER COMPARISON TABLE
# ============================================================

println("\n--- Schrödinger prediction,  L_ε = ", L_eps, " ---\n")

sl.interleaved ||
    @warn "parity sectors do not interleave — a mode may be missed or unresolved"

println(
    "kept ",
    length(measured) + 1,
    " real eigenvalues of ",
    length(λ_sorted),
    "; the remaining eigenvalues are complex and are not included in this comparison\n",
)

println(
    rpad("j", 4),
    rpad("parity", 8),
    rpad("ω_j", 20),
    rpad("predicted λ", 24),
    rpad("measured λ", 24),
    rpad("rel. eig. err", 18),
    "rel. gap err",
)

for j in 1:n_cmp
    λ_pred = predicted[j]
    λ_meas = Float64(measured[j])

    eig_rel_err = abs(λ_pred - λ_meas) / abs(λ_meas)

    gap_pred = 1 - λ_pred
    gap_meas = 1 - λ_meas

    gap_rel_err = abs(gap_pred - gap_meas) / abs(gap_meas)

    println(
        rpad(j - 1, 4),
        rpad(string(sl.parities[j]), 8),
        rpad(round(sl.eigenvalues[j]; sigdigits=10), 20),
        rpad(round(λ_pred; sigdigits=16), 24),
        rpad(round(λ_meas; sigdigits=16), 24),
        rpad(round(eig_rel_err; sigdigits=3), 18),
        round(gap_rel_err; sigdigits=3),
    )
end


# ============================================================
# PLOT 6 — Spectrum near λ = 1 with Schrödinger predictions
# ============================================================
# Blue points are the numerically computed transfer-operator spectrum.
# Red x's are the Schrödinger predictions, which lie on the real axis.
#
# Adjust X_ZOOM and Y_ZOOM below to control how tightly the plot is
# focused around λ = 1.

X_ZOOM = 0.00025
Y_ZOOM = 0.00015

λ_re_zoom = Float64.(real.(λ_sorted))
λ_im_zoom = Float64.(imag.(λ_sorted))
pred_plot = Float64.(predicted)

fig6, ax6 = subplots(figsize=(7, 5))

ax6.scatter(
    λ_re_zoom,
    λ_im_zoom;
    color="steelblue",
    s=25,
    zorder=3,
    label="computed spectrum",
)

ax6.scatter(
    pred_plot,
    zeros(length(pred_plot));
    color="firebrick",
    marker="x",
    s=80,
    linewidths=2.0,
    zorder=4,
    label="Schrödinger prediction",
)

ax6.axhline(
    0.0;
    color="dimgray",
    linewidth=0.8,
    alpha=0.6,
)

ax6.set_xlim(
    1 - X_ZOOM,
    1 + 0.05 * X_ZOOM,
)

ax6.set_ylim(
    -Y_ZOOM,
    Y_ZOOM,
)

ax6.set_xlabel(raw"Re($\lambda$)")
ax6.set_ylabel(raw"Im($\lambda$)")
ax6.set_title("Spectrum Near λ = 1  (α=$(ALPHA), ε=$(EPSILON))")
ax6.legend()
ax6.grid(true, alpha=0.3)

fig6.tight_layout()


# ============================================================
# REDUCED RESOLVENT NORM AT z = 1
# ============================================================
# I - P_ε is singular on all of L² because the invariant density lies in
# its kernel. The relevant inverse is therefore the restriction to the
# zero-mass subspace V_{N,0}:
#
#     M_ε
#       = ‖[(I - P_{ε,N})|_{V_{N,0}}]⁻¹‖_{L²→L²}
#       = 1 / σ_min(B₀).
#
# Internally:
#
#   1. (M, G) are transformed to L²-orthonormal coordinates;
#   2. B₀ represents I - P_{ε,N} restricted to the zero-mass subspace;
#   3. σ_min is computed by a dense SVD.
#
# For comparison with the resolvent at z = 1, define
#
#     d_ε = min_{j≥2} |1 - λ_j|,
#
# the distance from 1 to the nontrivial computed spectrum.
#
# This is distinct from the usual modulus spectral gap
#
#     1 - max_{j≥2} |λ_j|.
#
# The product d_ε M_ε is useful diagnostically. If it remains bounded, the
# approach of the spectrum to z = 1 accounts for the resolvent blow-up,
# without additional asymptotically growing non-normal amplification.
#
# mass_error checks whether the restricted problem still represents a
# mass-preserving operator. Exact mass preservation gives q*B = 0.
# A defect above mass_tolerance(T) means the resulting reduced-resolvent
# norm should not be trusted. The defect is reported rather than projected
# away.

rn = ResolventNorm.reduced_resolvent_norm(M, G, spls)

dist_1 = minimum(
    abs.(one(T) .- λ_sorted[2:end]),
)

println("\n--- reduced resolvent at z = 1 ---")
println("M_ε = 1/σ_min       = ", rn.norm)
println("σ_min, σ_max        = ", rn.sigma_min, ", ", rn.sigma_max)
println("dist(1, spectrum)   = ", dist_1)
println("dist · M_ε          = ", dist_1 * rn.norm)

println(
    "mass defect         = ",
    rn.mass_error,
    "   (tolerance ",
    ResolventNorm.mass_tolerance(T),
    ")",
)

println("partition of unity  = ", rn.pou_error)
println("κ(M), κ(M_s)        = ", rn.cond_mass, ", ", rn.cond_mass_scaled)


# ============================================================
# DISPLAY ALL FIGURES
# ============================================================
# Use pyplot.show() once, after all figures have been constructed, so that
# Matplotlib starts and manages the GUI event loop in one place.

show()