#-------------------------------------------
# Tests for IntervalHelpers.jl
#-------------------------------------------

using PmSpectrum.Utils: union_intervals, intersect_intervals!,
    mod_interval, split_circular!, overlaps_any

    
# Helpers: approximate equality for intervals and vectors of intervals, with a tolerance for floating-point fuzz.

function intervals_approx_equal(A::Tuple{T,T}, B::Tuple{T,T}; atol::T=eps(T)) where {T}
    abs(A[1]-B[1]) ≤ atol && abs(A[2]-B[2]) ≤ atol
end

function intervals_approx_equal(A::Vector{Tuple{T,T}}, B::Vector{Tuple{T,T}}; atol::T=eps(T)) where {T}

    length(A) == length(B) || return false
    for (x,y) in zip(A,B)
        abs(x[1]-y[1]) ≤ atol || return false
        abs(x[2]-y[2]) ≤ atol || return false
    end

    return true
end


#####################################################
# Tests for interval utility functions
#####################################################

@testset "union_intervals (non-wrapping)" begin
# Merge overlaps
x = [(0.1, 0.2), (0.15, 0.3), (0.5, 0.6)]
@test union_intervals(x) == [(0.1, 0.3), (0.5, 0.6)]

# Merge touching (within eps)
y = [(0.1, 0.2), (0.2, 0.25)]
@test union_intervals(y) == [(0.1, 0.25)]

# Already disjoint and sorted
z = [(0.0, 0.1), (0.2, 0.3)]
@test union_intervals(z) == z

# Empty input
@test union_intervals(Tuple{Float64,Float64}[]) == Tuple{Float64,Float64}[]
end


# Call the mutating version and return the filled `out`
function _run_intersect(I, J; atol=nothing)
    T = typeof(first(I)[1])
    out = NTuple{2,T}[]
    if atol === nothing
        intersect_intervals!(I, J; out=out)
    else
        intersect_intervals!(I, J; out=out, atol=T(atol))
    end
    return out
end

@testset "intersect_intervals!" begin
    # Basic overlap
    I = [(0.1, 0.4), (0.6, 0.9)]
    J = [(0.2, 0.3), (0.5, 0.7)]
    @test _run_intersect(I, J) == [(0.2, 0.3), (0.6, 0.7)]

    # Touching at a point → treated as empty (width ~ 0)
    I2 = [(0.1, 0.2)]
    J2 = [(0.2, 0.3)]
    @test _run_intersect(I2, J2) == Tuple{Float64,Float64}[]

    # Disjoint
    I3 = [(0.0, 0.1)]
    J3 = [(0.2, 0.3)]
    @test isempty(_run_intersect(I3, J3))

    # BigFloat cases with tiny overlaps around eps(BigFloat)
    setprecision(BigFloat, 256) do
        T      = BigFloat
        epsT   = eps(T)
        atolT  = 10epsT  # for approx comparison only

        I4 = [(T(0.1), T(0.2))]

        # overlap width smaller than epsT → should be discarded
        J4 = [(T(0.2) - epsT/2, T(0.25))]
        @test isempty(_run_intersect(I4, J4))  # default atol = eps(T)

        # overlap width larger than epsT → kept
        J5 = [(T(0.2) - 2epsT, T(0.25))]
        r  = _run_intersect(I4, J5)
        expected = [(T(0.2) - 2epsT, T(0.2))]
        @test intervals_approx_equal(r, expected; atol=atolT)
    end
end

@testset "mod_interval" begin
    T = Float64

    # --- 1. Basic mapping within [0,1]
    @test mod_interval((T(0.2), T(0.5))) == (T(0.2), T(0.5))
    @test mod_interval((T(0.0), T(0.3))) == (T(0.0), T(0.3))
    @test mod_interval((T(0.7), T(1.0))) == (T(0.7), T(1.0))

    # --- 2. Intervals that start/end at 0 or 1
    @test intervals_approx_equal(mod_interval((T(1.0), T(1.2))), (T(0.0), T(0.2)))
    @test intervals_approx_equal(mod_interval((T(-0.2), T(0.0))), (T(0.8), T(1.0)))

    # --- 3. Values outside [0,1] but within (-1,2)
    @test intervals_approx_equal(mod_interval((T(-0.2), T(-0.1))), (T(0.8), T(0.9)))
    @test intervals_approx_equal(mod_interval((T(1.2), T(1.5))), (T(0.2), T(0.5)))

    # --- 5. Error cases
    @test_throws ArgumentError mod_interval((T(0.3), T(0.3)))  # degenerate
    @test_throws ArgumentError mod_interval((T(-0.2), T(0.2))) # crosses 0
    @test_throws ArgumentError mod_interval((T(0.8), T(1.2)))  # crosses 1
    @test_throws ArgumentError mod_interval((T(-2.0), T(0.0))) # below -1
    @test_throws ArgumentError mod_interval((T(0.0), T(2.1)))  # above 2
end


@testset "split_circular! (symmetric windows, ε < 1/2, strict a<b)" begin
    T = Float64

    # 0) Tiny → empty (ensures we never emit a==b)
    parts = Tuple{T,T}[]
    split_circular!(parts, T(0.5), T(0.0))           # 2ε == 0 ≤ tol_empty
    @test isempty(parts)

    # 1) No wrap: interior
    parts = Tuple{T,T}[]
    split_circular!(parts, T(0.5), T(0.1))
    @test intervals_approx_equal(parts, [(T(0.4), T(0.6))])

    # 2) No wrap: touching left seam (a==0 but still a<b)
    ε = T(0.2)
    parts = Tuple{T,T}[]
    split_circular!(parts, ε, ε)                     # (0, 2ε)
    @test intervals_approx_equal(parts, [(T(0.0), T(0.4))])

    # 3) No wrap: touching right seam (b==1 but a<b)
    ε = T(0.15)
    parts = Tuple{T,T}[]
    split_circular!(parts, T(1) - ε, ε)             # (1-2ε, 1)
    @test intervals_approx_equal(parts, [(T(0.7), T(1.0))])

    # 4) Wrap-left: t < ε
    t, ε = T(0.05), T(0.1)                          # a = -0.05, b = 0.15
    parts = Tuple{T,T}[]
    split_circular!(parts, t, ε)
    @test intervals_approx_equal(parts, [(T(0.0), T(0.15)), (T(0.95), T(1.0))])

    # 5) Wrap-right: 1 - t < ε
    t, ε = T(0.97), T(0.05)                          # a = 0.92, b = 1.02
    parts = Tuple{T,T}[]
    split_circular!(parts, t, ε)
    @test intervals_approx_equal(parts, [(T(0.0), T(0.02)), (T(0.92), T(1.0))])

    # 6) Seam-centered tiny window (ε > 0, emits two strict-length pieces)
    t, ε = T(0.0), T(1e-6)
    parts = Tuple{T,T}[]
    split_circular!(parts, t, ε; atol=T(0))     # ensure not pruned
    @test intervals_approx_equal(parts, [(T(0.0), T(1e-6)), (T(1.0 - 1e-6), T(1.0))])

    # 7) Mutating append behavior and strictness invariant
    base = [(T(0.0), T(0.1))]
    ret  = split_circular!(base, T(0.5), T(0.1))
    @test ret === base
    @test length(base) == 2
    @test intervals_approx_equal(ret, [(T(0.0), T(0.1)), (T(0.4), T(0.6))])

end

@testset "overlaps_any" begin
    # --- single-tuple dispatch ---

    # clear overlap
    @test  overlaps_any(0.0, 0.5, (0.3, 0.8))
    # clear disjoint (left)
    @test !overlaps_any(0.0, 0.2, (0.3, 0.8))
    # clear disjoint (right)
    @test !overlaps_any(0.9, 1.0, (0.3, 0.8))
    # touching endpoints are not overlapping (strict inequality)
    @test !overlaps_any(0.0, 0.3, (0.3, 0.8))
    @test !overlaps_any(0.8, 1.0, (0.3, 0.8))
    # one interval contained in the other
    @test  overlaps_any(0.4, 0.6, (0.3, 0.8))
    @test  overlaps_any(0.1, 0.9, (0.3, 0.8))

    # --- collection dispatch ---

    intervals = [(0.1, 0.3), (0.5, 0.7), (0.9, 1.0)]
    # overlaps first
    @test  overlaps_any(0.0, 0.2, intervals)
    # overlaps middle
    @test  overlaps_any(0.6, 0.65, intervals)
    # overlaps last
    @test  overlaps_any(0.95, 1.0, intervals)
    # falls in a gap between first and second
    @test !overlaps_any(0.35, 0.45, intervals)
    # empty collection
    @test !overlaps_any(0.0, 1.0, NTuple{2,Float64}[])
end