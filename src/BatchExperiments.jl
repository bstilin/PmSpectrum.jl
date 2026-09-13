using Dates
using Printf
using UUIDs
using JLD2


# ─────────────────────────────────────────────────────────────────────────────
# Manifest entry
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExperimentEntry

Lightweight metadata record for one completed experiment. Stored in the
manifest so results can be filtered without loading the full JLD2 files.
Note that the metadata fields are redundant with the contents of the JLD2 files; they are
stored seperately to allow filtering and indexing without loading the full results. Thus,
no accuracy is lost due to downcasting to Float64 for the manifest.

Fields
------
- `filename`         : basename of the result file (UUID + ".jld2"), relative
                       to the campaign `output_dir`
- `date`             : ISO 8601 datetime string when the run completed
- `precision`        : numeric type used ("Float64" or "Double64")
- `alpha`            : PM map exponent (stored as Float64 for uniform indexing)
- `epsilon`          : uniform noise half-width (stored as Float64)
- `degree`           : B-spline degree
- `num_break_points` : breakpoints in [x_c, 1/2] passed to `build_pm_basis`
- `num_quad_points`  : Gauss–Legendre nodes per subinterval of S*
- `decouple`         : if `true`, the basis was built so that no B-spline support straddles
                       L_ε or 1-L_ε; each spline's support is entirely contained in [
                       [L_ε, 1-L_ε] or its complement
"""
struct ExperimentEntry
    filename         :: String
    date             :: String
    precision        :: String
    alpha            :: Float64
    epsilon          :: Float64
    degree           :: Int
    num_break_points :: Int
    num_quad_points  :: Int
    decouple         :: Bool
end


# ─────────────────────────────────────────────────────────────────────────────
# Manifest I/O
# ─────────────────────────────────────────────────────────────────────────────

const MANIFEST_FILE = "manifest.jld2"

function _manifest_path(output_dir::AbstractString)
    joinpath(output_dir, MANIFEST_FILE)
end

"""
    load_manifest(output_dir) -> Vector{ExperimentEntry}

Load the experiment index from `output_dir`. Returns an empty vector if no
manifest exists yet.
"""
function load_manifest(output_dir::AbstractString) :: Vector{ExperimentEntry}
    path = _manifest_path(output_dir)
    isfile(path) || return ExperimentEntry[]

    jldopen(path, "r") do f
        n = f["n"] :: Int
        n == 0 && return ExperimentEntry[]
        [ExperimentEntry(
            f["filenames"][i],
            f["dates"][i],
            f["types"][i],
            f["alphas"][i],
            f["epsilons"][i],
            f["degrees"][i],
            f["num_break_points"][i],
            f["num_quad_points"][i],
            f["decouples"][i],
        ) for i in 1:n]
    end
end

function _save_manifest(output_dir::AbstractString, entries::Vector{ExperimentEntry})
    mkpath(output_dir)
    jldopen(_manifest_path(output_dir), "w") do f
        f["n"]                = length(entries)
        f["filenames"]        = [e.filename         for e in entries]
        f["dates"]            = [e.date             for e in entries]
        f["types"]            = [e.precision        for e in entries]
        f["alphas"]           = [e.alpha            for e in entries]
        f["epsilons"]         = [e.epsilon          for e in entries]
        f["degrees"]          = [e.degree           for e in entries]
        f["num_break_points"] = [e.num_break_points for e in entries]
        f["num_quad_points"]  = [e.num_quad_points  for e in entries]
        f["decouples"]        = [e.decouple         for e in entries]
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

"""
    filter_results(entries; alpha, epsilon, degree, num_break_points,
                            num_quad_points, T, decouple, date_after, date_before)
                 -> Vector{ExperimentEntry}

Return the subset of `entries` matching all supplied keyword filters.
Any keyword left as `nothing` is treated as "match anything".

Continuous fields (`alpha`, `epsilon`) are matched with `≈` (relative tolerance
`1e-8`) so that floating-point roundtrip in the manifest doesn't cause misses.
`alpha`/`epsilon` may be passed as decimal strings (e.g. `"1e-3"`, as stored in
`ALPHAS`/`EPSILONS` grids) and are parsed to `Float64` before comparison.
Integer, boolean, and string fields are matched exactly.
`date_after` / `date_before` are ISO 8601 strings; lexicographic comparison is
exact for the `"yyyy-mm-ddTHH:MM:SS"` format written by `run_grid`.
"""
function filter_results(
    entries :: Vector{ExperimentEntry};
    alpha            = nothing,
    epsilon          = nothing,
    degree           = nothing,
    num_break_points = nothing,
    num_quad_points  = nothing,
    T                = nothing,
    decouple         = nothing,
    date_after       = nothing,
    date_before      = nothing,
) :: Vector{ExperimentEntry}

    _match_approx(val, target::Nothing) = true
    _match_approx(val, target::AbstractString) = isapprox(val, parse(Float64, target); rtol=1e-8)
    _match_approx(val, target) = isapprox(val, Float64(target); rtol=1e-8)

    _match_exact(val, target::Nothing) = true
    _match_exact(val, target) = val == target

    filter(entries) do e
        _match_approx(e.alpha,   alpha)   &&
        _match_approx(e.epsilon, epsilon) &&
        _match_exact(e.degree,           degree)           &&
        _match_exact(e.num_break_points, num_break_points) &&
        _match_exact(e.num_quad_points,  num_quad_points)  &&
        _match_exact(e.precision, T === nothing ? nothing : string(T)) &&
        _match_exact(e.decouple, decouple) &&
        (date_after  === nothing || e.date >= string(date_after))  &&
        (date_before === nothing || e.date <= string(date_before))
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Loading results
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_results(output_dir, entries) -> Vector{InvariantDensityResult}

Load the full `InvariantDensityResult` for each entry in `entries`.
The numeric type is inferred from the `"type"` key stored in each file. 
See save_result in Results.jl
"""
function load_results(
    output_dir :: AbstractString,
    entries    :: Vector{ExperimentEntry},
) :: Vector{InvariantDensityResult}
    [load_result(joinpath(output_dir, e.filename)) for e in entries]
end


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_single_experiment(alpha, epsilon, num_break_points, num_quad_points;
                          degree=3, decouple=false, left_inverse=nothing,
                          refine=Bases.EndpointRefinement(alpha))
        -> InvariantDensityResult{T}

Run a single invariant-density computation and return the result.

Arguments
---------
- `alpha`            : PM map exponent α ∈ (0, 1)
- `epsilon`          : noise half-width 0 < ε < 1/2
- `num_break_points` : breakpoints in [x_c, 1/2] passed to `build_pm_basis`
- `num_quad_points`  : Gauss–Legendre nodes per subinterval of S*
- `degree`           : B-spline degree (default 3) For standard use with this package, DO NOT change this.
                        not all features will work as expected with other degrees.
- `decouple`         : if `true`, no B-spline support straddles L_ε or 1-L_ε; each spline is entirely
                       contained in [L_ε, 1-L_ε] or its complement (default false)
- `left_inverse`     : optional prebuilt inverse of the left branch (from
                       `Utils.construct_left_branch_inverse(alpha)`). The inverse depends only on
                       `(alpha, T)`, so callers sweeping ε or the mesh (e.g. `run_grid`) can build it
                       once per alpha and pass it here to skip the expensive BigFloat table build.
- `refine`           : dyadic endpoint refinement of the S* quadrature partition (see
                       `Bases.EndpointRefinement`). Defaults to `Bases.EndpointRefinement(alpha)`
                       with τ_end = `Bases.default_tau_end(T)`; pass `nothing` to disable.
"""
function run_single_experiment(
    alpha            :: T,
    epsilon          :: T,
    num_break_points :: Int,
    num_quad_points  :: Int;
    degree           :: Int  = 3,
    decouple         :: Bool = false,
    left_inverse            = nothing,
    refine           :: Union{Nothing,Bases.EndpointRefinement{T}} = Bases.EndpointRefinement(alpha),
) :: InvariantDensityResult{T} where {T<:AbstractFloat}

    T === Float64 || T === Double64 ||
        throw(ArgumentError("Unsupported type T = $T. Only Float64 and Double64 are supported."))

    # Basis
    B, _ = Bases.build_pm_basis(alpha, epsilon, num_break_points, T;
                                p=degree, decouple=decouple)
    break_points = unique!(collect(T, BSplineKit.knots(B)))

    # Branches (tuple, not vector: b1 and b2 have different concrete types, and
    # tuple iteration keeps branch.fwd calls statically dispatched in assembly)
    # Both halves of b2 are the reflection of b1's: `symmetric_pm` gives x = 1/2
    # to the left branch, so b2 needs the right-branch formula to satisfy the
    # `Branch` endpoint contract (b2.fwd(0.5) == 0, not 1).
    left_inv  = left_inverse === nothing ? Utils.construct_left_branch_inverse(alpha) : left_inverse
    right_inv = x -> one(T) - left_inv(one(T) - x)
    right_fwd = x -> one(T) - Utils.symmetric_pm(one(T) - x, alpha)
    b1 = Bases.Branch(x -> Utils.symmetric_pm(x, alpha), left_inv,
                      (zero(T), T(0.5)), (zero(T), one(T)))
    b2 = Bases.Branch(right_fwd, right_inv,
                      (T(0.5),  one(T)), (zero(T), one(T)))

    # Galerkin matrices (single-spline wrappers built once and shared)
    spls = Bases.build_single_splines(B)
    M = Bases.mass_matrix(spls)
    G = Bases.transfer_matrix(spls, spls, epsilon, (b1, b2);
                                  n_quad=num_quad_points, atol=eps(T), refine=refine)

    λ_raw, V_raw = Utils.scaled_nonsymmetric_eigen(G, M)
    p        = sortperm(abs.(λ_raw); rev=true)
    λ_sorted = λ_raw[p]
    V_sorted = V_raw[:, p]

    # Sanity diagnostics on the leading eigenpair (warnings only, never throw)
    λ1 = λ_sorted[1]
    abs(imag(λ1)) > sqrt(eps(T)) * abs(λ1) &&
        @warn "Leading eigenvalue has a non-negligible imaginary part" λ1 alpha epsilon
    abs(λ1 - 1) > T(1e-6) &&
        @warn "Leading eigenvalue deviates from 1 by more than 1e-6" λ1 alpha epsilon

    # Normalize first eigenvector (invariant density) to unit L¹ mass
    c    = real.(V_sorted[:, 1])
    mass = sum(c[i] * spls[i].mass for i in eachindex(c))
    c    = c / mass

    minimum(c) < -T(1e-6) * maximum(c) &&
        @warn "Invariant-density coefficients are significantly negative" minimum(c) alpha epsilon

    InvariantDensityResult{T}(
        alpha, epsilon, degree, num_break_points, num_quad_points,
        decouple,
        break_points,
        M, G,
        real.(λ_sorted),
        imag.(λ_sorted),
        c,
    )
end


# ─────────────────────────────────────────────────────────────────────────────
# Grid runner
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_grid(output_dir, alphas, epsilons, num_break_points_list,
             num_quad_points_list; degree=3, T=Float64, decouple=false,
             overwrite=false, verbose=true)

Run all combinations of the supplied parameter vectors and save each result
to `output_dir`. A manifest index is maintained at `output_dir/manifest.jld2`
so results can be queried later with `load_manifest` / `filter_results`.

If `overwrite=false` (the default), any combination already present in the
manifest is skipped. Set `overwrite=true` to recompute and overwrite.

Arguments
---------
- `output_dir`           : directory to store results (created if absent)
- `alphas`               : vector of α values, either `Real`s or decimal strings
- `epsilons`             : vector of ε values, either `Real`s or decimal strings
- `num_break_points_list`: vector of num_break_points values
- `num_quad_points_list` : vector of num_quad_points values
- `degree`               : B-spline degree (default 3)
- `T`                    : numeric type, `Float64` or `Double64` (default `Float64`)
- `decouple`             : `Bool` or vector of `Bool`s. If `true`, no B-spline support
                           straddles L_ε or 1-L_ε; each spline is entirely contained in
                           [L_ε, 1-L_ε] or its complement. Pass `[false, true]` to run both
                           variants. (default `false`)
- `overwrite`            : recompute existing combinations (default `false`)
- `verbose`              : print progress (default `true`)

Passing `alphas`/`epsilons` as decimal strings (e.g. `"1e-7"`) rather than
`Float64` literals ensures each value is parsed directly in the target
precision `T` via `parse(T, ...)`. 
"""
function run_grid(
    output_dir              :: AbstractString,
    alphas                  :: AbstractVector,
    epsilons                :: AbstractVector,
    num_break_points_list   :: AbstractVector{Int},
    num_quad_points_list    :: AbstractVector{Int};
    degree    :: Int                            = 3,
    T         :: Type{<:AbstractFloat}          = Float64,
    decouple  :: Union{Bool,AbstractVector{Bool}} = false,
    overwrite :: Bool                           = false,
    verbose   :: Bool                           = true,
)
    mkpath(output_dir)
    entries   = load_manifest(output_dir)
    decouples = decouple isa Bool ? [decouple] : decouple

    total = length(alphas) * length(epsilons) *
            length(num_break_points_list) * length(num_quad_points_list) *
            length(decouples)
    done  = 0

    _as_T(x::AbstractString) = parse(T, x)
    _as_T(x::Real)           = T(x)

    # The left-branch inverse depends only on (alpha, T); build once per alpha
    # instead of once per experiment (each build is a 5000-knot BigFloat table).
    inverse_cache = Dict{T,Any}()

    for alpha_raw in alphas, epsilon_raw in epsilons,
            nbp in num_break_points_list, nqp in num_quad_points_list,
            dc in decouples

        alpha   = _as_T(alpha_raw)
        epsilon = _as_T(epsilon_raw)
        done += 1

        # Check for existing entry
        existing = filter_results(entries;
            alpha=alpha, epsilon=epsilon, degree=degree,
            num_break_points=nbp, num_quad_points=nqp, T=T, decouple=dc)

        if !isempty(existing) && !overwrite
            verbose && @info "[$done/$total] Skipping (already exists)" alpha epsilon nbp nqp dc
            continue
        end

        verbose && @info "[$done/$total] Running" alpha epsilon nbp nqp T dc

        left_inverse = get!(() -> Utils.construct_left_branch_inverse(alpha),
                            inverse_cache, alpha)

        result = run_single_experiment(alpha, epsilon, nbp, nqp;
                                       degree=degree, decouple=dc,
                                       left_inverse=left_inverse)

        filename = string(uuid4()) * ".jld2"
        save_result(joinpath(output_dir, filename), result)

        entry = ExperimentEntry(
            filename,
            Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            string(T),
            Float64(alpha),
            Float64(epsilon),
            degree, nbp, nqp, dc,
        )

        # Remove any old entry for same params if overwriting
        if overwrite && !isempty(existing)
            old_files = Set(e.filename for e in existing)
            filter!(e -> e.filename ∉ old_files, entries)
        end

        push!(entries, entry)
        _save_manifest(output_dir, entries)
    end

    verbose && @info "Done. $(length(entries)) total results in $output_dir"
    return entries
end


# ─────────────────────────────────────────────────────────────────────────────
# Manifest table
# ─────────────────────────────────────────────────────────────────────────────

"""
    show_manifest(entries; alpha, epsilon, degree, num_break_points,
                           num_quad_points, T, decouple, date_after, date_before,
                           sort_by) -> Vector{ExperimentEntry}

Pretty-print the manifest as a filtered, sorted ASCII table and return the
matching entries. Accepts the same keyword filters as `filter_results`.

`sort_by` is a tuple of field symbols controlling the sort order (default
`(:precision, :alpha, :epsilon, :num_break_points)`). Any subset of
`(:precision, :alpha, :epsilon, :degree, :num_break_points, :num_quad_points, :decouple, :date)`
is valid; `:T` is accepted as a legacy alias for `:precision`.

## Example
```julia
manifest = load_manifest("data/3-30-26")
entries  = show_manifest(manifest; alpha=0.5, T=Double64)
```
"""
function show_manifest(
    entries :: Vector{ExperimentEntry};
    alpha            = nothing,
    epsilon          = nothing,
    degree           = nothing,
    num_break_points = nothing,
    num_quad_points  = nothing,
    T                = nothing,
    decouple         = nothing,
    date_after       = nothing,
    date_before      = nothing,
    sort_by          :: Tuple = (:precision, :alpha, :epsilon, :num_break_points),
) :: Vector{ExperimentEntry}

    filtered = filter_results(entries;
        alpha=alpha, epsilon=epsilon, degree=degree,
        num_break_points=num_break_points, num_quad_points=num_quad_points,
        T=T, decouple=decouple, date_after=date_after, date_before=date_before)

    _field(e, s) = getfield(e, s === :T ? :precision : s)   # :T = legacy alias
    sorted = sort(filtered; by = e -> Tuple(_field(e, s) for s in sort_by))

    # ── header ────────────────────────────────────────────────────────────────
    cols   = ("T", "alpha", "epsilon", "degree", "nbp", "nqp", "decouple", "date")
    widths = (8,   8,       12,        6,         6,     5,     8,          19)

    sep = join(("-"^w for w in widths), "-+-") * "-"
    hdr = join(lpad(c, w) for (c, w) in zip(cols, widths))

    println()
    println("  ", hdr)
    println("  ", sep)

    for e in sorted
        row = (
            rpad(e.precision, widths[1]),
            lpad(@sprintf("%.4f",   e.alpha),        widths[2]),
            lpad(@sprintf("%.3e",   e.epsilon),       widths[3]),
            lpad(string(e.degree),                    widths[4]),
            lpad(string(e.num_break_points),          widths[5]),
            lpad(string(e.num_quad_points),           widths[6]),
            lpad(string(e.decouple),                  widths[7]),
            lpad(e.date[1:min(19,end)],               widths[8]),
        )
        println("  ", join(row, " | "))
    end

    println("  ", sep)
    println("  $(length(sorted)) / $(length(entries)) entries")
    println()

    return sorted
end
