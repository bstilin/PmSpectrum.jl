# =============================================================================
#   Analysis of the sweep produced by `tutorial_parameter_sweep.jl`.
#
#   Loads every epsilon for ONE fixed precision and ONE fixed alpha, then plots:

#     PLOT 1  invariant densities at several noise amplitudes
#     PLOT 2  collapse to similarity solution profile U_0 of those densities
#     PLOT 3  L1 convergence as epsilon -> 0
#     PLOT 4  spectral gap closure as epsilon -> 0
#     PLOT 5  second eigenfunction on the circle
#     PLOT 6  second eigenfunction near the fixed point
#     PLOT 7  boundary layer collapse of the second eigenfunction
#
#   Everything is loaded once at the top into `runs`, and every plot below
#   reuses those saved results. After `include` you can poke at `runs`,
#   `epsilons`, `phi2s`, ... in the REPL and redraw any single plot by itself.
#
#   NOISE AMPLITUDE CONVENTION:
#   The numerics use a kick uniform on [-epsilon, epsilon], so the stored
#   epsilon is a HALF-WIDTH. The paper writes the kick as epsilon*xi with xi of
#   unit variance, so the paper's epsilon is a STANDARD DEVIATION:
#
#       eps_paper = eps_stored / sqrt(3).
#
#   Every epsilon on an axis or in a legend below is the PAPER's. The filenames,
#   the EPSILON_TAGS list, and Utils.boundary_layer_scale all keep the stored
#   half-width. L_eps is a physical length and is the same number either way;
#   the scaling exponents are also unchanged, since a constant factor on epsilon
#   moves a power law's prefactor but not its slope.
#
# COMMAND-LINE USAGE:
#
#       julia --project=. scripts/analyzing_tutorial_sweep.jl
#
#   Run `scripts/tutorial_parameter_sweep.jl` first; this script reads what that
#   one wrote, and the parameters below must match the ones it ran.
#
# OUTPUT:
#
#   plots/tutorial_plots/
#
#   Each figure is written as a .pdf, with alpha, precision and resolution in
#   the filename, and is also left open on screen.
# =============================================================================

# ENV["MPLBACKEND"] = "tkagg"  # for interactive plotting on some platforms

using PyPlot
using PmSpectrum
using BSplineKit
using DoubleFloats
using Printf

close("all")


# ============================================================
# PARAMETERS — this is the part you edit
# ============================================================
# The first block selects which saved runs to load. ALPHA_TAG and EPSILON_TAGS
# stay DECIMAL STRINGS because the filenames are built from them; numerical
# versions are parsed or read from the loaded results further down.

FLOAT_TYPE       = Float64   # must be a precision the sweep actually ran

ALPHA_TAG        = "0.5"     # one alpha per run of this script

EPSILON_TAGS     = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6",
                    "1e-7", "1e-8", "1e-9", "1e-10", "1e-11"]

NUM_BREAK_POINTS = 150
NUM_QUAD_POINTS  = 64

# Which epsilon PLOT 5 and PLOT 6 should show. This is matched by filename tag,
# so its meaning does not change if EPSILON_TAGS is reordered or files are
# missing.
PHI_EPSILON_TAG = "1e-5"


# Evaluation-grid and plotting ranges that are useful to tweak in experiments.
GRID_POINTS  = 2001
GRID_DECADES = -14

DENSITY_COLLAPSE_T_MIN    = 1e-2
DENSITY_COLLAPSE_T_MAX    = 20.0
DENSITY_COLLAPSE_T_POINTS = 400

PHI_COLLAPSE_T_MIN    = 1e-2
PHI_COLLAPSE_T_MAX    = 1e6
PHI_COLLAPSE_T_POINTS = 600

SLOPE_GUIDE_OFFSET = 0.70


# ============================================================
# PATHS AND SMALL UTILITIES
# ============================================================
# @__DIR__ is the directory containing this script. Moving one level up gives
# the repository root, so output paths do not depend on where Julia is launched.
# Sweep data is saved to data/tutorial_sweep and plots to plots/tutorial_plots.

REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
DATA_DIR  = joinpath(REPO_ROOT, "data", "tutorial_sweep")
PLOTS_DIR = joinpath(REPO_ROOT, "plots", "tutorial_plots")

# savefig writes a file but does not create directories.
mkpath(PLOTS_DIR)

# Every figure is written as <number>_<name>_<TAG>, so runs at different alpha
# or basis resolution land side by side instead of overwriting.
TAG = "alpha$(ALPHA_TAG)_$(FLOAT_TYPE)_nbp$(NUM_BREAK_POINTS)_nqp$(NUM_QUAD_POINTS)"

# Keep the repeated save mechanics in one place while leaving each plot's actual
# construction explicit below. Each figure is written twice: the PDF is the
# vector copy to read or drop into a document, and the PNG is what Markdown can
# actually display, which is how the README figures are produced.
function save_figure(fig, name)
    fig.tight_layout()
    path = joinpath(PLOTS_DIR, name)
    fig.savefig("$path.pdf")
    fig.savefig("$path.png"; dpi=200)
end


# ============================================================
# LOAD THE SWEEP
# ============================================================
# A missing file is reported and skipped rather than fatal, so a sweep that was
# interrupted part way through can still be analyzed.
#
# Each element of `runs` keeps the filename epsilon tag, stored result, and
# rehydrated basis/density together so they cannot accidentally get out of sync.

runs = []

println("=== Loading alpha = ", ALPHA_TAG, ", ", FLOAT_TYPE, " from ", DATA_DIR, " ===")

for eps_tag in EPSILON_TAGS

    filename =
        "alpha$(ALPHA_TAG)_eps$(eps_tag)_$(FLOAT_TYPE)" *
        "_nbp$(NUM_BREAK_POINTS)_nqp$(NUM_QUAD_POINTS).jld2"

    path = joinpath(DATA_DIR, filename)

    if !isfile(path)
        println("missing  ", filename)
        continue
    end

    result = load_result(path)

    push!(runs, (
        eps_tag=eps_tag,
        result=result,
        rehydrated=rehydrate(result),
    ))

    println("loaded   ", filename)
end

isempty(runs) && error(
    "no files found in $DATA_DIR. Run scripts/tutorial_parameter_sweep.jl " *
    "first, and check that ALPHA_TAG, FLOAT_TYPE, NUM_BREAK_POINTS and " *
    "NUM_QUAD_POINTS above match the sweep that wrote the files."
)

# Give the rest of the script one deterministic ordering, independent of the
# order in EPSILON_TAGS: largest stored epsilon first, smallest last.
sort!(runs; by=run -> Float64(run.result.epsilon), rev=true)


# ============================================================
# DERIVED QUANTITIES
# ============================================================

alpha = parse(FLOAT_TYPE, ALPHA_TAG)

# Stored half-widths, and the boundary layer scale L_eps = (eps^2/6)^(1/(2+a)).
# boundary_layer_scale takes the HALF-WIDTH, so it is fed the stored epsilon.
epsilons = Float64[Float64(run.result.epsilon) for run in runs]

L_eps = Float64[
    Float64(Utils.boundary_layer_scale(run.result.alpha, run.result.epsilon))
    for run in runs
]

# The paper's epsilon. Used for display only, never for selecting a run.
EPS_SCALE = 1 / sqrt(3)
eps_paper = EPS_SCALE .* epsilons

# The run whose epsilon is smallest stands in for the epsilon -> 0 limit.
i_ref = argmin(epsilons)

# Select the PLOT 5 / PLOT 6 run by its filename epsilon tag rather than by
# position, so missing files or a reordered EPSILON_TAGS list cannot change it.
PHI_INDEX = findfirst(run -> run.eps_tag == PHI_EPSILON_TAG, runs)
isnothing(PHI_INDEX) && error(
    "PHI_EPSILON_TAG = $PHI_EPSILON_TAG was not loaded. " *
    "Choose one of: $(join((run.eps_tag for run in runs), ", "))"
)

println("\nloaded ", length(runs), " runs")
println("smallest epsilon is ", runs[i_ref].eps_tag, " (used as the reference below)")
println("PLOT 5 / PLOT 6 epsilon is ", runs[PHI_INDEX].eps_tag)

# One color per epsilon, shared by PLOT 1, PLOT 2 and PLOT 7 so that a given
# epsilon keeps its color across all three. Viridis is useful because the family
# is ordered from large epsilon to small epsilon. Stopping at 0.85 keeps the
# curves off the bright yellow end, which is hard to see on white.
eps_colors = [
    PyPlot.get_cmap("viridis")((k - 1) / max(length(runs) - 1, 1) * 0.85)
    for k in eachindex(runs)
]


# ============================================================
# EVALUATION GRID
# ============================================================
# Log spaced and symmetric about 1/2, so it is dense at BOTH neutral fixed
# points, x = 0 and x = 1.

xs = Utils.symmetric_logspace_grid(
    GRID_POINTS;
    decades=GRID_DECADES,
    include_mid=true,
    include_ends=false,
    T=FLOAT_TYPE,
)

# The left half alone, for everything that looks at the layer at x = 0.
xs_half = filter(x -> x <= FLOAT_TYPE(0.5), xs)


# ============================================================
# PLOT 1 — Invariant densities at several noise amplitudes
# ============================================================
# Solid lines are the computed invariant densities. The x markers are the
# boundary-layer prediction.
#
# The prediction is an expansion about the fixed point at x = 0, so it is drawn
# only on x <= 1/2 and only on log axes. It is intended to describe the local
# boundary-layer structure, not the bulk of the density or the corresponding
# layer near x = 1.
#
# We expect the x markers to lie increasingly close to the solid curves as
# epsilon decreases. At the same time, the boundary layer should become
# narrower and the peak at x = 0 should grow taller, approaching the singular
# noiseless profile as epsilon -> 0.

xs_marker = xs_half[1:60:end]   # thin the markers so they stay readable

fig, ax = subplots(figsize=(7, 5))

for (k, run) in pairs(runs)

    ax.plot(
        Float64.(xs),
        Float64.(run.rehydrated.pdf.(xs));
        color=eps_colors[k],
        linewidth=1.5,
        label=@sprintf("ε = %.1e", eps_paper[k]),
    )

    perturb = Utils.construct_perturbation_solution(
        run.result.alpha,
        run.result.epsilon,
        run.rehydrated.pdf(FLOAT_TYPE(0.5));
        order=:second,
    )

    ax.plot(
        Float64.(xs_marker),
        Float64.(perturb.(xs_marker));
        color=eps_colors[k],
        linestyle="none",
        marker="x",
        markersize=6,
    )
end

# An empty series, purely so the marker gets a legend entry of its own.
ax.plot(
    Float64[],
    Float64[];
    color="dimgray",
    linestyle="none",
    marker="x",
    markersize=6,
    label="boundary layer prediction",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("x")
ax.set_ylabel(raw"$\rho_\varepsilon(x)$")
ax.set_title("Invariant Density vs. Noise Amplitude  (α=$(ALPHA_TAG))")
ax.legend(loc="lower left", fontsize=7, ncol=2)
ax.grid(true, alpha=0.3)

save_figure(fig, "1_invariant_density_$(TAG)")


# ============================================================
# PLOT 2 — Boundary layer collapse of the densities
# ============================================================
# According to the formal asymptotic analysis, each density should satisfy
#
#     rho_eps(x) = U_eps(t) * L_eps^(-alpha),      t = x / L_eps,
#
# so plotting U_eps(t) = rho_eps(t L_eps) * L_eps^alpha against t should
# collapse the whole family onto the epsilon -> 0 similarity solution
#
#     U_0(t) = K2 * E(2/beta, (a/beta) t^beta),
#
# with beta = alpha + 2 and a = 2^alpha. The amplitude K2 is set by rho_0(1/2),
# which is approximated from the invariant density for the smallest epsilon loaded.
#
# This plot tests both profile collapse and the predicted similarity shape.
# Even when the inner shapes collapse, a residual amplitude drift can remain
# because rho_eps(1/2) approaches its limiting value only asymptotically.

beta    = Float64(alpha) + 2.0
a_const = exp2(Float64(alpha))

ts = exp10.(range(
    log10(DENSITY_COLLAPSE_T_MIN),
    log10(DENSITY_COLLAPSE_T_MAX);
    length=DENSITY_COLLAPSE_T_POINTS,
))

fig2, ax2 = subplots(figsize=(7, 5))

for (k, run) in pairs(runs)

    U = Float64[
        Float64(run.rehydrated.pdf(FLOAT_TYPE(t * L_eps[k]))) * L_eps[k]^Float64(alpha)
        for t in ts
    ]

    ax2.plot(
        ts,
        U;
        color=eps_colors[k],
        linewidth=1.5,
        label=@sprintf("ε = %.1e", eps_paper[k]),
    )
end

K2 = Float64(runs[i_ref].rehydrated.pdf(FLOAT_TYPE(0.5))) / beta^2 * (beta / a_const)^(2 / beta)

U0 = Float64[
    K2 * Utils.exp_times_upper_gamma(2 / beta, (a_const / beta) * t^beta)
    for t in ts
]

ax2.plot(
    ts,
    U0;
    color="black",
    linestyle="none",
    marker="x",
    markevery=0.05,
    label=raw"$U_0(t)$",
)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(raw"$t = x \,/\, L_\varepsilon$")
ax2.set_ylabel(raw"$\rho_\varepsilon(t L_\varepsilon)\,L_\varepsilon^{\alpha}$")
ax2.set_title("Collapse of the Invariant Density  (α=$(ALPHA_TAG))")
ax2.legend(fontsize=8, ncol=2)
ax2.grid(true, alpha=0.3)

save_figure(fig2, "2_density_collapse_$(TAG)")


# ============================================================
# PLOT 3 — L1 convergence
# ============================================================
# ||rho_eps - rho_ref||_1 against epsilon, with the reference density taken
# from the smallest epsilon in the sweep. The predicted rate is
#
#     zeta = 2(1 - alpha) / (2 + alpha).
#
# The dashed line is drawn AT that slope and offset from the data. The formal 
# asymptotic analysis predicts this slope, and the data should run parallel to it.
#
# Near the reference epsilon, finite-reference effects can bend the curve away
# from the asymptotic guide: rho_ref is itself an epsilon > 0 computation rather
# than the true epsilon -> 0 limit. 
#
# Each epsilon has its own adapted mesh, so the two-basis form of
# l1_norm_difference is the one that applies.

eps_l1 = Float64[]
l1     = Float64[]

for (k, run) in pairs(runs)

    k == i_ref && continue

    push!(eps_l1, eps_paper[k])

    push!(l1, Float64(Bases.l1_norm_difference(
        run.rehydrated.basis,
        run.result.c,
        runs[i_ref].rehydrated.basis,
        runs[i_ref].result.c;
        atol=eps(FLOAT_TYPE),
    )))
end

zeta = 2 * (1 - Float64(alpha)) / (2 + Float64(alpha))

# Anchored on the first point, pushed below the data by SLOPE_GUIDE_OFFSET.
xs_guide3 = [minimum(eps_l1), maximum(eps_l1)]
ys_guide3 = SLOPE_GUIDE_OFFSET * l1[1] .* (xs_guide3 ./ eps_l1[1]) .^ zeta

fig3, ax3 = subplots(figsize=(7, 5))

ax3.plot(
    eps_l1,
    l1;
    color="steelblue",
    linestyle="none",
    marker="o",
    markersize=6,
    label=raw"$\|\rho_\varepsilon - \rho_{\rm ref}\|_1$",
)

ax3.plot(
    xs_guide3,
    ys_guide3;
    color="firebrick",
    linestyle="--",
    linewidth=1.5,
    label="predicted slope $(round(zeta; digits=3))",
)

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(raw"$\varepsilon$")
ax3.set_ylabel(raw"$\|\rho_\varepsilon - \rho_{\rm ref}\|_1$")
ax3.set_title("L¹ Convergence  (α=$(ALPHA_TAG))")
ax3.legend()
ax3.grid(true, alpha=0.3)

save_figure(fig3, "3_l1_convergence_$(TAG)")


# ============================================================
# PLOT 4 — Spectral gap closure
# ============================================================
# Plot 1 - |lambda_2| against epsilon using the stored eigenvalues.
#
# The boundary-layer Schrödinger problem predicts that the leading
# nonstationary eigenvalue is real. Numerically, however, we compute the gap
# using |lambda_2| so that the diagnostic remains well-defined even if a small
# imaginary part appears from discretization or roundoff.
#
# The predicted scaling exponent is
#
#     s = 2 alpha / (2 + alpha),
#
# and the dashed reference line is drawn with this predicted slope rather than
# fitted to the computed data.
#
# At alpha = 1/2 this exponent and PLOT 3's zeta are both 0.4. That equality is
# specific to alpha = 1/2; changing ALPHA_TAG separates the two predicted rates.

gap = Float64[
    Float64(1 - hypot(run.result.eigenvalues_re[2], run.result.eigenvalues_im[2]))
    for run in runs
]

s = 2 * Float64(alpha) / (2 + Float64(alpha))

xs_guide4 = [minimum(eps_paper), maximum(eps_paper)]
ys_guide4 = SLOPE_GUIDE_OFFSET * gap[1] .* (xs_guide4 ./ eps_paper[1]) .^ s

fig4, ax4 = subplots(figsize=(7, 5))

ax4.plot(
    eps_paper,
    gap;
    color="steelblue",
    linestyle="none",
    marker="o",
    markersize=6,
    label=raw"$1 - |\lambda_2|$",
)

ax4.plot(
    xs_guide4,
    ys_guide4;
    color="firebrick",
    linestyle="--",
    linewidth=1.5,
    label="predicted slope $(round(s; digits=3))",
)

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel(raw"$\varepsilon$")
ax4.set_ylabel(raw"$1 - |\lambda_2|$")
ax4.set_title("Spectral Gap Closure  (α=$(ALPHA_TAG))")
ax4.legend()
ax4.grid(true, alpha=0.3)

save_figure(fig4, "4_spectral_gap_$(TAG)")


# ============================================================
# SECOND EIGENFUNCTION
# ============================================================
# The sweep saved only the leading eigenvector, but it saved M and G in full, so
# phi_2 is recovered exactly by re-solving G v = lambda M v on the stored
# matrices. Nothing is reassembled and nothing is re-quadratured.
#
# An eigenvector has no intrinsic scale, so a convention is unavoidable. Here
# every phi_2 is divided by its SIGNED value at the fixed point, which sets
#
#     phi_2(0) = 1


phi2s = []

println("\n--- recovering φ₂ (re-solving G v = λ M v on the stored matrices) ---")

for (k, run) in pairs(runs)

    λ, V = Utils.scaled_nonsymmetric_eigen(run.result.G, run.result.M)

    p  = sortperm(abs.(λ); rev=true)
    λ2 = λ[p[2]]

    abs(imag(λ2)) > sqrt(eps(FLOAT_TYPE)) * abs(λ2) && @warn(
        "λ₂ is part of a complex pair at ε = $(run.eps_tag); its real part " *
        "alone is not an eigenfunction, so that curve is not meaningful."
    )

    c2 = real.(V[:, p[2]])
    φ2 = BSplineKit.Spline(run.rehydrated.basis, c2)

    φ0 = φ2(zero(FLOAT_TYPE))

    push!(phi2s, BSplineKit.Spline(run.rehydrated.basis, c2 ./ φ0))

    println(
        "ε = ", rpad(run.eps_tag, 6),
        "   λ₂ = ", λ2,
        "   φ₂(0) = ", Float64(φ0), " (divided out)",
    )
end


# ============================================================
# PLOT 5 — Second eigenfunction on the circle
# ============================================================
# The map lives on the circle, so [0,1] is shifted to [-1/2,1/2] by
#
#     x -> x        for x <= 1/2,
#     x -> x - 1    otherwise,
#
# which glues the two endpoints into the single neutral fixed point at 0. The
# log spaced grid is dense near x = 0 and x = 1, hence dense on both sides of
# the origin after the shift.
#
# phi_2 changes sign, so its positive and negative parts are drawn as two
# separate nonnegative curves on a log axis. Where a part vanishes its curve
# just breaks, because the zeros are stored as NaN.

xc   = [x <= FLOAT_TYPE(0.5) ? Float64(x) : Float64(x) - 1.0 for x in xs]
perm = sortperm(xc)

φ_circle = Float64[Float64(phi2s[PHI_INDEX](x)) for x in xs][perm]

φp = [v > 0 ?  v : NaN for v in φ_circle]
φm = [v < 0 ? -v : NaN for v in φ_circle]

fig5, ax5 = subplots(figsize=(7, 5))

ax5.plot(
    xc[perm],
    φp;
    color="steelblue",
    linewidth=1.5,
    label=raw"$\varphi_+$",
)

ax5.plot(
    xc[perm],
    φm;
    color="firebrick",
    linestyle="--",
    linewidth=1.5,
    label=raw"$\varphi_-$",
)

ax5.set_yscale("log")
ax5.set_xlim(-0.5, 0.5)
ax5.set_xlabel("x")
ax5.set_ylabel(raw"$\varphi_\pm(x)$")
ax5.set_title(@sprintf("Second Eigenfunction on the Circle  (α=%s, ε=%.1e)",
                       ALPHA_TAG, eps_paper[PHI_INDEX]))
ax5.legend()
ax5.grid(true, alpha=0.3)

save_figure(fig5, "5_phi2_circle_eps$(runs[PHI_INDEX].eps_tag)_$(TAG)")


# ============================================================
# PLOT 6 — Second eigenfunction near the fixed point
# ============================================================
# The same run on log-log axes over x in (0, 1/2].

φ_half = Float64[Float64(phi2s[PHI_INDEX](x)) for x in xs_half]

φp_half = [v > 0 ?  v : NaN for v in φ_half]
φm_half = [v < 0 ? -v : NaN for v in φ_half]

fig6, ax6 = subplots(figsize=(7, 5))

ax6.plot(
    Float64.(xs_half),
    φp_half;
    color="steelblue",
    linewidth=1.5,
    label=raw"$\varphi_+$",
)

ax6.plot(
    Float64.(xs_half),
    φm_half;
    color="firebrick",
    linestyle="--",
    linewidth=1.5,
    label=raw"$\varphi_-$",
)

ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.set_xlabel(raw"$x \in (0,\, 1/2]$")
ax6.set_ylabel(raw"$\varphi_\pm(x)$")
ax6.set_title(@sprintf("Second Eigenfunction Near the Fixed Point  (α=%s, ε=%.1e)",
                       ALPHA_TAG, eps_paper[PHI_INDEX]))
ax6.legend()
ax6.grid(true, alpha=0.3)

save_figure(fig6, "6_phi2_fixed_point_eps$(runs[PHI_INDEX].eps_tag)_$(TAG)")


# ============================================================
# PLOT 7 — Boundary-layer collapse of the second eigenfunction
# ============================================================
# This plot tests whether the second eigenfunction develops a universal
# boundary-layer profile on the same spatial scale L_eps as the invariant
# density.
#
# For each epsilon, phi_2 has already been normalized so that
#
#     phi_2(0) = 1.
#
# We then introduce the inner coordinate
#
#     t = x / L_eps
#
# and plot
#
#     Phi_eps(t) = phi_2(t L_eps).
#
# Because of the normalization at x = 0, every rescaled curve satisfies
# Phi_eps(0) = 1. Thus this plot tests collapse of the PROFILE SHAPE only;
# no additional amplitude rescaling is applied.
#
# If L_eps is the correct spatial scale for the second eigenfunction, the
# rescaled curves should approach a common profile as epsilon decreases.
#
# Solid curves show the positive part of phi_2, while dashed curves show the
# magnitude of the negative part. The zero crossing need not occur on the
# same scale as the positive inner profile, so failure of the zero crossings
# to collapse does not by itself rule out boundary-layer collapse near x = 0.
#
# The t window extends far enough to display the negative branch after the
# sign change. For each epsilon, the curve is stopped once x = t L_eps reaches
# 1/2, so the plot remains restricted to the half interval containing the
# boundary layer at x = 0.

ts7 = exp10.(range(
    log10(PHI_COLLAPSE_T_MIN),
    log10(PHI_COLLAPSE_T_MAX);
    length=PHI_COLLAPSE_T_POINTS,
))

fig7, ax7 = subplots(figsize=(7, 5))

for (k, run) in pairs(runs)

    Φ = Float64[
        t * L_eps[k] <= 0.5 ?
            Float64(phi2s[k](FLOAT_TYPE(t * L_eps[k]))) : NaN
        for t in ts7
    ]

    ax7.plot(
        ts7,
        [v > 0 ? v : NaN for v in Φ];
        color=eps_colors[k],
        linewidth=1.5,
        label=@sprintf("ε = %.1e", eps_paper[k]),
    )

    ax7.plot(
        ts7,
        [v < 0 ? -v : NaN for v in Φ];
        color=eps_colors[k],
        linestyle="--",
        linewidth=1.5,
    )
end

# Empty series so the dashed style gets its own legend entry.
ax7.plot(
    Float64[],
    Float64[];
    color="dimgray",
    linestyle="--",
    linewidth=1.5,
    label=raw"negative part $-\varphi_2$",
)

ax7.set_xscale("log")
ax7.set_yscale("log")
ax7.set_xlabel(raw"$t = x \,/\, L_\varepsilon$")
ax7.set_ylabel(raw"$|\Phi_\varepsilon(t)| = |\varphi_2(tL_\varepsilon)|$")
ax7.set_title("Boundary-Layer Collapse of the Second Eigenfunction  (α=$(ALPHA_TAG))")
ax7.legend(fontsize=8, ncol=2)
ax7.grid(true, alpha=0.3)

name7 = "7_phi2_collapse_$(TAG)"

save_figure(fig7, name7)

# ============================================================
# DISPLAY ALL FIGURES
# ============================================================
# One show() at the end, after every figure has been built, so that Matplotlib
# starts and manages the GUI event loop in a single place.

show()