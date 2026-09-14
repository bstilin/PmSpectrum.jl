# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

# =============================================================================
# tutorial_parameter_sweep.jl
#
# PURPOSE:
#   Compute the spectrum of the transfer operator of the noisy symmetric Pomeau-Manneville map
#   for every combination of
#
#       alpha  x  epsilon  x  precision (Float64 / Double64)
#
#   listed in the parameter block below, save each result to its own JLD2 file,
#   and then load one back. The loop is written out in the open so you can see
#   exactly what one "run" is; edit the parameter lists, re-run, done.
#
#   This is the starting point for learning the package. Its production
#   counterpart is `run_grid` (src/BatchExperiments.jl), which does the same
#   sweep but adds a manifest index, UUID filenames, and the
#   `load_manifest` / `filter_results` query tooling. Once your sweeps outgrow
#   "I can find the file by its name", switch to that. 
#
#   OUTPUT DIRECTORY:
#   This script finds the repository root from its own location using @__DIR__.
#   Since this file lives in `scripts/`,
#
#       @__DIR__                  -> <repo>/scripts
#       joinpath(@__DIR__, "..")  -> <repo>
#
#   Results are therefore always written to
#
#       <repo>/data/tutorial_sweep/
#
#   regardless of the directory from which Julia is launched. `mkpath` creates
#   this directory automatically if it does not already exist.
#
#   WHICH KNOBS COST YOU TIME:
#     NUM_BREAK_POINTS  size of the basis. The transfer matrix is dense in the
#                       number of splines, so doubling this is roughly 4x the
#                       assembly and ~8x the eigensolve. The dominant knob.
#
#     NUM_QUAD_POINTS   quadrature nodes per subinterval of composite gaussian
#                       quadrature rule. Cost is linear here.
#
#     PRECISIONS        Double64 is much slower than Float64 on two counts:
#                       every arithmetic operation is emulated in double-double,
#                       and the eigensolve falls back to a generic Schur
#                       decomposition instead of LAPACK. Start with Float64.
#
#     EPSILONS          small epsilon is not itself slow, but it needs a finer
#                       basis to resolve the boundary layer, which is.
#
#   The script prints a wall-clock time per run. Ignore the FIRST
#   time of each precision: it is mostly Julia compiling the code, not solving
#   the problem. At the default settings the first run may take several seconds
#   and subsequent runs are substantially faster.
#
# COMMAND-LINE USAGE:
#
#   From the repository root:
#
#       julia --project=. scripts/tutorial_parameter_sweep.jl
#
#   The `--project=.` flag tells Julia to activate the Project.toml in the
#   repository root, so `using PmSpectrum` loads this checkout of the package
#   and its declared dependencies.
#
#   Because output paths are based on @__DIR__, not the current working
#   directory, the data files are always placed under the repository's `data/`
#   directory.
#
# OUTPUT:
#
#   data/tutorial_sweep/
#
#   One .jld2 file is written per run, with the parameters encoded in the
#   filename.
# =============================================================================

using PmSpectrum


# ============================================================
# PATHS
# ============================================================
# @__DIR__ is the directory containing this script:
#
#     <repo>/scripts
#
# Going up one directory gives the repository root. Building paths from this
# location means output does not depend on where the user launched Julia.

REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
OUTPUT_DIR = joinpath(REPO_ROOT, "data", "tutorial_sweep")

# `save_result` writes a file but does not create directories, so make sure the
# output directory exists before starting the sweep.
mkpath(OUTPUT_DIR)


# ============================================================
# PARAMETERS — this is the part you edit
# ============================================================

# alpha and epsilon are DECIMAL STRINGS, not Float64 literals. Each is parsed
# directly in the target precision below, so `parse(Double64, "1e-3")` resolves
# the true decimal to ~32 digits. Writing `Double64(1e-3)` instead would only
# widen a value that had already been rounded to Float64.
#
# The strings also serve as exact filename tags.

ALPHAS   = ["0.25", "0.5", "0.75"]   # PM map exponent, 0 < alpha < 1

EPSILONS = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6",
            "1e-7","1e-8", "1e-9", "1e-10", "1e-11"]  # noise half-width, 0 < epsilon < 1/2

PRECISIONS = [Float64]  # try [Float64, Double64] -- see header

NUM_BREAK_POINTS = 150   # breakpoints in [x_c, 1/2] of the hybrid mesh
NUM_QUAD_POINTS  = 64   # Gauss-Legendre nodes per subinterval of S*

# false: a re-run skips any combination whose file already exists, so an
# interrupted sweep picks up where it stopped.
#
# true: recompute and overwrite everything.
OVERWRITE = false


# ============================================================
# WHERE EACH RESULT GOES
# ============================================================
# The filename itself is the index: every parameter that distinguishes one run
# appears in it.
#
# Example:
#
#     data/tutorial_sweep/alpha0.5_eps1e-3_Float64_nbp60_nqp16.jld2

function result_path(T, alpha_str, eps_str)
    filename =
        "alpha$(alpha_str)_eps$(eps_str)_$(T)" *
        "_nbp$(NUM_BREAK_POINTS)_nqp$(NUM_QUAD_POINTS).jld2"

    return joinpath(OUTPUT_DIR, filename)
end


# ============================================================
# THE SWEEP
# ============================================================
# A note on failures:
#
# `run_single_experiment` never throws merely because a numerical result is
# poor; it warns. If an @warn about the leading eigenvalue or negative density
# coefficients scrolls past, that run likely did not converge. Usually this
# means the basis is too coarse for that epsilon. Increase NUM_BREAK_POINTS
# and retry.

n_runs = length(PRECISIONS) * length(ALPHAS) * length(EPSILONS)

println(
    "=== Sweep: ",
    length(PRECISIONS),
    " x ",
    length(ALPHAS),
    " x ",
    length(EPSILONS),
    " = ",
    n_runs,
    " runs ===",
)

println("output directory: ", OUTPUT_DIR, "\n")


for T in PRECISIONS
    for alpha_str in ALPHAS

        alpha = parse(T, alpha_str)

        # The inverse of the left branch needs a 5000-knot BigFloat warm start table which
        # depends only on (alpha, T), never on epsilon.
        #
        # Build it once here and pass it into every epsilon run rather than
        # rebuilding it each time.
        left_inverse = Utils.construct_left_branch_inverse(alpha)

        for eps_str in EPSILONS

            # alpha and epsilon must have the SAME concrete type. That type is
            # how run_single_experiment selects its working precision.
            epsilon = parse(T, eps_str)
            path = result_path(T, alpha_str, eps_str)

            if isfile(path) && !OVERWRITE
                println("skip   ", basename(path))
                continue
            end

            elapsed = @elapsed result = run_single_experiment(
                alpha,
                epsilon,
                NUM_BREAK_POINTS,
                NUM_QUAD_POINTS;
                left_inverse=left_inverse,
            )

            save_result(path, result)

            # Eigenvalues are stored sorted by |lambda|, descending, with real
            # and imaginary parts separated.
            #
            # lambda_1 should equal 1 to close to working precision. lambda_2 need not be
            # real, so the modulus spectral gap is
            #
            #     1 - |lambda_2|.
            #
            # hypot(re, im) computes |lambda_2| without assuming it is real.

            lambda_1 = result.eigenvalues_re[1]

            gap = 1 - hypot(
                result.eigenvalues_re[2],
                result.eigenvalues_im[2],
            )

            # Printed as Float64 because these are only screen diagnostics.
            # The saved JLD2 file retains the full working precision.
            #
            # The first run of each precision includes compilation time, so
            # compare timings only after the first run.

            println(
                "saved  ",
                basename(path),
                "   lambda_1 = ",
                Float64(lambda_1),
                "   1-|lambda_2| = ",
                Float64(gap),
                "   (",
                round(elapsed; digits=1),
                " s)",
            )
        end
    end
end


