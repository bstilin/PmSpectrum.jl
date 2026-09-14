# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

###############################################################################
# Integration of observables against the invariant density
###############################################################################

"""
    build_integration_mesh(base_knots::Vector{T}, ε::T;
                           atol::T = eps(T),
                           refine::Union{Nothing,EndpointRefinement{T}} = nothing) ::Vector{T}

Quadrature partition of [0,1] for integrating against a spline density.

This is `build_star_partition` specialised to the identity branch on [0,1], so
the partition is exactly

1. the endpoints 0 and 1;
2. `base_knots ∩ [0,1]` (deduplicated to tolerance `atol`);
3. the shifted breakpoints `mod(p ± ε, 1)` for every `p` in (2) — the identity
   branch's `inv` is a no-op, so no tabulated map inverse is needed here;
4. if `refine` is given, the dyadic (ratio-2) geometric grid at both endpoints,
    (see `EndpointRefinement`).

sorted and deduplicated.

Note on (3): the ±ε shifts are *not* required for ∫ f ρ — a spline density is
smooth across them — but they cost nothing, and they make the same mesh correct
for integrands that involve `G_ε` (which is piecewise polynomial with
breakpoints at the shifted knots).

Layer (4) is the one that matters for observables: ρ is a piecewise polynomial
and is therefore resolved exactly by (1)-(3), so all of the quadrature error
comes from `f`.  For an observable with a *stronger* singularity than the map's
`x^α`, deepen the grid with `EndpointRefinement(α; τ_end = ...)`; the dyadic
depth is `N_end = ceil(log2(1/τ_end) / (2 + α))`.
"""
function build_integration_mesh(
    base_knots :: Vector{T},
    ε          :: T;
    atol       :: T = eps(T),
    refine     :: Union{Nothing,EndpointRefinement{T}} = nothing,
) ::Vector{T} where {T<:AbstractFloat}

    (zero(T) <= ε < one(T)/T(2)) ||
        throw(ArgumentError("ε must be in [0, 0.5), got $ε"))

    id = Branch(identity, identity, (zero(T), one(T)), (zero(T), one(T)))
    return build_star_partition(id, base_knots, ε; atol=atol, refine=refine)
end


"""
    _gl_rule(::Type{T}, n_quad::Int) -> (nodes::Vector{T}, weights::Vector{T})

Gauss-Legendre rule on [-1,1], generated in `BigFloat` and cast to `T` so that
the nodes and weights carry full `T` accuracy (same idiom as `transfer_matrix`).
"""
function _gl_rule(::Type{T}, n_quad::Int) where {T<:AbstractFloat}
    n_quad >= 1 || throw(ArgumentError("n_quad must be ≥ 1, got $n_quad"))
    ξ, ω = legendre(BigFloat, n_quad)
    return T.(ξ), T.(ω)
end


"""
    integrate_on_mesh(f, mesh::Vector{T}; n_quad::Int = 32, atol::T = eps(T)) ::T

Composite Gauss-Legendre quadrature of `f` over the partition `mesh`, with
`n_quad` nodes per panel.  Panels narrower than `atol` are skipped.

Density-free counterpart of `integrate_against_density`; useful for observables
alone, and for checking a mesh against integrals with known values.
"""
function integrate_on_mesh(
    f,
    mesh   :: AbstractVector{T};
    n_quad :: Int = 32,
    atol   :: T   = eps(T),
) ::T where {T<:AbstractFloat}
    nodes, weights = _gl_rule(T, n_quad)
    return Utils._gl_integrate_mesh(f, mesh, nodes, weights; atol=atol)
end


"""
    DensityQuadrature{T}

A quadrature rule with the invariant density folded into the weights, so that

    ∫₀¹ f(x) ρ(x) dx  ≈  Σ_q w[q] · f(x[q]).

Build one per density and reuse it across observables: ρ is evaluated once per
node at construction, so each additional observable costs only its own
evaluations.

Fields
------
- `mesh` : the partition from `build_integration_mesh`
- `x`    : quadrature nodes, flattened over all kept panels
- `w`    : `half_k · ω_q · ρ(x_q)` — the density is already included
"""
struct DensityQuadrature{T<:AbstractFloat}
    mesh :: Vector{T}
    x    :: Vector{T}
    w    :: Vector{T}
end

Base.length(dq::DensityQuadrature) = length(dq.x)

function Base.show(io::IO, dq::DensityQuadrature{T}) where {T}
    print(io, "DensityQuadrature{", T, "}(", length(dq.mesh) - 1, " panels, ",
          length(dq.x), " nodes, mass = ", total_mass(dq), ")")
end


"""
    density_quadrature(pdf, mesh::Vector{T}; n_quad::Int = 32, atol::T = eps(T))
    density_quadrature(pdf, base_knots::Vector{T}, ε::T;
                       n_quad::Int = 32, atol::T = eps(T), refine = nothing)

Build a `DensityQuadrature` for the density callable `pdf`.

The second form builds the mesh first via `build_integration_mesh`; the first
takes a mesh you already have (e.g. shared between several densities on the
same basis).

`pdf` is any callable accepting a `T` — typically a `BSplineKit.Spline`, but a
plain function works too (pass `x -> one(T)` to get the bare quadrature rule).
"""
function density_quadrature(
    pdf,
    mesh   :: AbstractVector{T};
    n_quad :: Int = 32,
    atol   :: T   = eps(T),
) ::DensityQuadrature{T} where {T<:AbstractFloat}

    length(mesh) >= 2 || throw(ArgumentError("mesh needs at least two points"))
    issorted(mesh)    || throw(ArgumentError("mesh must be sorted"))

    nodes, weights = _gl_rule(T, n_quad)
    nq = length(nodes)

    # Count the panels that survive the atol filter, so x and w are sized once.
    nkeep = 0
    @inbounds for k in firstindex(mesh):(lastindex(mesh)-1)
        mesh[k+1] - mesh[k] > atol && (nkeep += 1)
    end

    x = Vector{T}(undef, nkeep * nq)
    w = Vector{T}(undef, nkeep * nq)

    idx = 0
    @inbounds for k in firstindex(mesh):(lastindex(mesh)-1)
        a, b = mesh[k], mesh[k+1]
        b - a > atol || continue
        half = (b - a) / T(2)
        cen  = a + half
        for q in 1:nq
            xq       = cen + half * nodes[q]
            idx     += 1
            x[idx]   = xq
            w[idx]   = (half * weights[q]) * pdf(xq)
        end
    end

    return DensityQuadrature{T}(collect(T, mesh), x, w)
end

function density_quadrature(
    pdf,
    base_knots :: Vector{T},
    ε          :: T;
    n_quad     :: Int = 32,
    atol       :: T   = eps(T),
    refine     :: Union{Nothing,EndpointRefinement{T}} = nothing,
) ::DensityQuadrature{T} where {T<:AbstractFloat}

    mesh = build_integration_mesh(base_knots, ε; atol=atol, refine=refine)
    return density_quadrature(pdf, mesh; n_quad=n_quad, atol=atol)
end


"""
    integrate_against_density(f, dq::DensityQuadrature{T}) ::T

Evaluate `∫₀¹ f(x) ρ(x) dx` using a prebuilt `DensityQuadrature`.

`f` must accept a `T`; when `T = Double64`, write the observable so that its
constants stay in `T` (e.g. close over `α::Double64`) rather than silently
demoting to `Float64`.
"""
function integrate_against_density(f, dq::DensityQuadrature{T}) ::T where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for q in eachindex(dq.x)
        acc += dq.w[q] * f(dq.x[q])
    end
    return acc
end


"""
    integrate_against_density(f, pdf, base_knots::Vector{T}, ε::T;
                              n_quad = 32, atol = eps(T), refine = nothing) ::T

One-shot form: build the mesh and the rule, then integrate.  Prefer the
`DensityQuadrature` form when integrating several observables against the same
density.
"""
function integrate_against_density(
    f,
    pdf,
    base_knots :: Vector{T},
    ε          :: T;
    n_quad     :: Int = 32,
    atol       :: T   = eps(T),
    refine     :: Union{Nothing,EndpointRefinement{T}} = nothing,
) ::T where {T<:AbstractFloat}

    dq = density_quadrature(pdf, base_knots, ε; n_quad=n_quad, atol=atol, refine=refine)
    return integrate_against_density(f, dq)
end


"""
    total_mass(dq::DensityQuadrature{T}) ::T

`∫₀¹ ρ(x) dx` — the sum of the folded weights.  For a normalised invariant
density this should be 1 to round-off, so it doubles as a self-check on the
mesh and the rule.
"""
total_mass(dq::DensityQuadrature{T}) where {T<:AbstractFloat} = sum(dq.w)
