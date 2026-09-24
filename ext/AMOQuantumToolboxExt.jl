#

module AMOQuantumToolboxExt

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff

import AMO.TimeSequence: QobjEvoStep
using AMO.TimeSequence: TimeSequence, _to_op, _lrmul, _lmul
using QuantumToolbox: AbstractQuantumObject, QuantumObjectEvolution, QobjEvo,
    isoper, issuper
using SciMLOperators: AbstractSciMLOperator, AbstractSciMLScalarOperator, ScalarOperator,
    MatrixOperator, IdentityOperator, ScaledOperator, AddedOperator, update_coefficients,
    isconstant, cache_operator
using SciMLBase: ODEProblem, init, solve!, reinit!
using OrdinaryDiffEqVerner: Vern7

##### Forward-mode differentiation of the coefficients

struct _QTag end

@inline function _dual_params(p::SVector{K,PT}) where {K,PT}
    return SVector{K}(ntuple(Val(K)) do k
                          partials = ForwardDiff.Partials(ntuple(j->PT(j == k), Val(K)))
                          return ForwardDiff.Dual{_QTag}(p[k], partials)
                      end)
end
@inline _dual_value(x::Number) = x
@inline _dual_value(d::ForwardDiff.Dual) = ForwardDiff.value(d)
@inline _dual_value(z::Complex{<:ForwardDiff.Dual}) =
    Complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
@inline _dual_partial(x::Number, k) = zero(x)
@inline _dual_partial(d::ForwardDiff.Dual, k) = ForwardDiff.partials(d, k)
@inline _dual_partial(z::Complex{<:ForwardDiff.Dual}, k) =
    Complex(ForwardDiff.partials(real(z), k), ForwardDiff.partials(imag(z), k))

##### Decomposition of the SciMLOperator tree of a `QuantumObjectEvolution`

# Evaluate a scalar operator (a coefficient) at parameters `p` and time `t`.
# For a plain `ScalarOperator` (what `QobjEvo` builds from a coefficient function), call its
# update function directly rather than through `update_coefficients`, which would
# allocate a new `ScalarOperator` for every evaluation.
@inline _scalar_value(λ::ScalarOperator, p, t) = λ.update_func(λ.val, nothing, p, t)
@inline _scalar_value(λ::AbstractSciMLScalarOperator, p, t) =
    convert(Number, update_coefficients(λ, nothing, p, t))
@inline _scalar_value(c::Number, p, t) = c
@inline _scalars_value(λs::Tuple, p, t) = prod(λ->_scalar_value(λ, p, t), λs)

# Collect the terms of the operator tree into constant matrices (`consts`) and
# time-dependent terms `(matrix, scalar operators)` (`tds`).
# `λs` is the tuple of scalar operators scaling the current sub-tree.
function _collect_terms!(consts, tds, op::MatrixOperator, λs)
    if !isconstant(op)
        throw(ArgumentError("MatrixOperator with an update function is not supported"))
    end
    return _push_term!(consts, tds, op.A, λs)
end
_collect_terms!(consts, tds, op::IdentityOperator, λs) =
    _push_term!(consts, tds, sparse(I, size(op, 1), size(op, 1)), λs)
_collect_terms!(consts, tds, op::ScaledOperator, λs) =
    _collect_terms!(consts, tds, op.L, (λs..., op.λ))
function _collect_terms!(consts, tds, op::AddedOperator, λs)
    for o in op.ops
        _collect_terms!(consts, tds, o, λs)
    end
    return
end
_collect_terms!(consts, tds, op::AbstractSciMLOperator, λs) =
    throw(ArgumentError("Unsupported operator of type $(typeof(op)) in the QuantumObjectEvolution"))

function _push_term!(consts, tds, A, λs)
    if all(isconstant, λs)
        # `p` is unused for constant coefficients
        c = _scalars_value((λs..., one(eltype(A))), nothing, zero(real(eltype(A))))
        push!(consts, c * A)
    else
        push!(tds, (A, λs))
    end
    return
end

##### Assembly of the generators
#
# All term matrices are stored by their values on the shared sparsity pattern `P`
# (the union of the patterns of all terms), so that assembling the generator
# `A(p, t) = H0 + Σᵢ cᵢ(p, t) Oᵢ` and its derivatives `∂ₖA = Σᵢ ∂ₖcᵢ Oᵢ` is a few
# vector operations, evaluating each coefficient function once (with dual numbers).
# Four `MatrixOperator`s with in-place updates are built, which are directly the right
# hand sides of the ODEs (the `-im` of the Schrödinger equation is folded into the values):
#   H_fwd: A(p, t) = -im H(p, t)            the generator (n × n)
#   H_rev: A(p, t0 + t1 - t)ᵀ               time-reversed and transposed, whose propagator
#                                           over [t0, t1] is Uᵀ (so that L U = (Uᵀ Lᵀ)ᵀ);
#                                           this holds for any generator, unitary or not
#   G_fwd, G_rev: the block lower triangular sensitivity systems ((K+1)n)
#       I ⊗ A + Σₖ |k⟩⟨0| ⊗ ∂ₖA
#   acting on [ψ; ∂₁ψ; ...; ∂ₖψ] for a single state ψ, stored as sparse matrices with
#   fixed structure into which the values of `A` and `∂ₖA` are scattered.

struct _Terms{T,PT,M,Λ<:NTuple{M,Tuple},PM}
    λss::Λ                   # scalar operators of the time-dependent terms
    param_map::PM
    H0nz::Vector{T}          # values of the constant term on `P`
    Onz::NTuple{M,Vector{T}} # values of the time-dependent terms on `P`
    tsum::PT                 # t0 + t1
    reverse::Bool            # evaluate the coefficients at `tsum - t`
end
@inline _time(terms::_Terms, t) = terms.reverse ? terms.tsum - t : t

# acc + Σⱼ cs[j] * Onz[j][q], unrolled over the terms
@inline _accum(acc, ::Tuple{}, ::Tuple{}, q) = acc
@inline _accum(acc, cs::Tuple, Onz::Tuple, q) =
    _accum(muladd(@inbounds(cs[1]), @inbounds(Onz[1][q]), acc), Base.tail(cs), Base.tail(Onz), q)

# Values of all the time-dependent coefficients at `p`, `t`
@inline _coefs(terms::_Terms, p, t) = map(λs->_scalars_value(λs, p, t), terms.λss)

# In-place update of the (n × n) generator: a single fused pass over the nonzeros
struct _PropUpdate{TT<:_Terms}
    terms::TT
end
function (f::_PropUpdate)(A, u, p, t)
    terms = f.terms
    cs = _coefs(terms, terms.param_map(p), _time(terms, t))
    nz = nonzeros(A)
    H0nz = terms.H0nz
    Onz = terms.Onz
    @inbounds for q in eachindex(nz)
        nz[q] = _accum(H0nz[q], cs, Onz, q)
    end
    return A
end

# In-place update of the sensitivity system. In the column-major storage of the block
# matrix `I ⊗ A + Σₖ |k⟩⟨0| ⊗ ∂ₖA`, the block columns 1:K only hold the diagonal block,
# whose values are exactly those of `A` in the order of the pattern: they are filled with
# contiguous copies of `Anz`. Block column 0 interleaves, for each column, the values of
# `A` and of the `∂ₖA`: they are written directly at the precomputed positions `dest0`
# and `destB[k]` of each nonzero of the pattern, in the same fused pass over the pattern
# that assembles the values.
struct _AugUpdate{TT<:_Terms,T,K}
    terms::TT
    Anz::Vector{T}          # values of A on the pattern
    dest0::Vector{Int}      # position in the block matrix of each value of A (block column 0)
    destB::NTuple{K,Vector{Int}} # the same for ∂ₖA
    diag_off::NTuple{K,Int} # start of the (contiguous) diagonal block of the block columns 1:K
end
function (f::_AugUpdate{TT,T,K})(M, u, p, t) where {TT,T,K}
    terms = f.terms
    cs = _coefs(terms, terms.param_map(_dual_params(p)), _time(terms, t))
    cv = map(_dual_value, cs)
    cp = ntuple(k->map(c->_dual_partial(c, k), cs), Val(K))
    _assemble_aug!(M, f, cv, cp)
    return M
end
function _assemble_aug!(M, f::_AugUpdate{TT,T,K}, cv, cp) where {TT,T,K}
    terms = f.terms
    nz = nonzeros(M)
    Anz = f.Anz
    H0nz = terms.H0nz
    Onz = terms.Onz
    dest0 = f.dest0
    destB = f.destB
    @inbounds for q in eachindex(Anz)
        a = _accum(H0nz[q], cv, Onz, q)
        Anz[q] = a
        nz[dest0[q]] = a
        for k in 1:K
            nz[destB[k][q]] = _accum(zero(T), cp[k], Onz, q)
        end
    end
    nnzP = length(Anz)
    @inbounds for k in 1:K
        copyto!(nz, f.diag_off[k], Anz, 1, nnzP)
    end
    return M
end

_pattern(A::AbstractMatrix) = sparse(A) .!= 0
_pattern(A::SparseMatrixCSC) = A .!= 0

# Values of `A` on the pattern `P`, in the storage order of `P`
function _values_on(::Type{T}, P::SparseMatrixCSC, A) where T
    rv = rowvals(P)
    vals = Vector{T}(undef, nnz(P))
    for c in 1:size(P, 2), q in nzrange(P, c)
        vals[q] = A[rv[q], c]
    end
    return vals
end

_with_values(P::SparseMatrixCSC, vals::Vector) =
    SparseMatrixCSC(size(P, 1), size(P, 2), copy(P.colptr), copy(P.rowval), vals)

function _make_prop(P, terms::_Terms, constant::Bool)
    A = _with_values(P, copy(terms.H0nz))
    return constant ? MatrixOperator(A) : MatrixOperator(A; update_func! = _PropUpdate(terms))
end

function _make_aug(Mpat, ::Type{T}, nnzP, K, terms::_Terms, constant::Bool, dest0, destB, diag_off) where T
    M = _with_values(Mpat, zeros(T, nnz(Mpat)))
    aug = _AugUpdate(terms, zeros(T, nnzP), dest0, Tuple(destB), Tuple(diag_off))
    # Initialize with the constant part (zero coefficients and derivatives)
    M0 = length(terms.Onz)
    _assemble_aug!(M, aug, ntuple(_->zero(T), M0), ntuple(_->ntuple(_->zero(T), M0), Val(K)))
    return constant ? MatrixOperator(M) : MatrixOperator(M; update_func! = aug)
end

function _build_operators(::Type{T}, ::Type{PT}, n, K, consts, tds, factor, param_map, tsum) where {T,PT}
    H_fwd, G_fwd = _build_pair(T, PT, n, K, consts, tds, factor, param_map, tsum, false)
    H_rev, G_rev = _build_pair(T, PT, n, K, [transpose(A) for A in consts],
                               [(transpose(A), λs) for (A, λs) in tds], factor, param_map,
                               tsum, true)
    return H_fwd, H_rev, G_fwd, G_rev
end

function _build_pair(::Type{T}, ::Type{PT}, n, K, consts, tds, factor, param_map, tsum,
                     reverse::Bool) where {T,PT}
    # Shared sparsity pattern
    P = spzeros(Bool, n, n)
    for A in consts
        P = P .| _pattern(A)
    end
    for (A, _) in tds
        P = P .| _pattern(A)
    end
    nnzP = nnz(P)
    Pidx = zeros(Int, n, n)
    rvP = rowvals(P)
    for c in 1:n, q in nzrange(P, c)
        Pidx[rvP[q], c] = q
    end

    H0nz = zeros(T, nnzP)
    for A in consts
        H0nz .+= factor .* _values_on(T, P, A)
    end
    Onz = Tuple(factor .* _values_on(T, P, A) for (A, _) in tds)
    λss = Tuple(λs for (_, λs) in tds)
    constant = isempty(tds)
    terms = _Terms(λss, param_map, H0nz, Onz, PT(tsum), reverse)

    # Sensitivity system structure and index maps
    Mpat = kron(sparse(I, K + 1, K + 1), P)
    for k in 1:K
        Ek = sparse([k + 1], [1], [true], K + 1, K + 1)
        Mpat = Mpat .| kron(Ek, P)
    end
    # Positions of the values in block column 0 (indexed by the nonzero of the pattern)
    dest0 = zeros(Int, nnzP)
    destB = [zeros(Int, nnzP) for _ in 1:K]
    rvM = rowvals(Mpat)
    for c in 1:n, q in nzrange(Mpat, c)
        r = rvM[q]
        br, rr = divrem(r - 1, n)
        s = Pidx[rr + 1, c]
        @assert s > 0
        if br == 0
            dest0[s] = q
        else
            destB[br][s] = q
        end
    end
    # The diagonal blocks of the block columns 1:K are contiguous copies of the pattern
    colptr = SparseArrays.getcolptr(Mpat)
    diag_off = [colptr[k * n + 1] for k in 1:K]
    @assert all(colptr[(k + 1) * n + 1] - colptr[k * n + 1] == nnzP for k in 1:K)

    H = _make_prop(P, terms, constant)
    G = _make_aug(Mpat, T, nnzP, K, terms, constant, dest0, destB, diag_off)
    return H, G
end

##### Work buffers and integrators
#
# The ODE problems for one column (state of size n) and for one column with its
# sensitivities (size (K+1)n), forward and reversed, are initialized once; each solve
# reinitializes the integrator with the new initial state and parameters. Buffers whose
# size depends on the number of columns `k` of a right multiplier or rows `m` of a left
# multiplier are (re)allocated lazily when that number changes.
mutable struct _QobjBuffers{T,IF,IR,IGF,IGR,IU,IY}
    const y0::Vector{T}     # initial state (n) for the propagation of one column
    const y0_aug::Vector{T} # initial state ((K+1)n) for the sensitivity system
    const int_fwd::IF       # integrators (forward/reversed, state/sensitivities)
    const int_rev::IR
    const int_gfwd::IGF
    const int_grev::IGR
    const U0::Matrix{T}     # identity, initial state for the full propagator (n × n)
    const Y0::Matrix{T}     # [I; 0; ...], initial state for the full sensitivities ((K+1)n × n)
    const int_full::IU      # matrix state integrators (one sparse × dense product per step)
    const int_gfull::IY
    k::Int
    UR::Matrix{T}           # U * R (n × k)
    S::Vector{Matrix{T}}    # ∂ₖU * R (n × k each)
    m::Int
    Lt::Matrix{T}           # Lᵀ (n × m)
    Y::Matrix{T}            # Uᵀ Lᵀ (n × m)
    DY::Vector{Matrix{T}}   # ∂ₖUᵀ Lᵀ (n × m each)
end
function _make_integrator(op, u0, tspan, params, alg, kws)
    prob = ODEProblem{true}(cache_operator(op, u0), u0, tspan, params)
    return init(prob, alg; save_everystep=false, save_start=false, save_end=false,
                dense=false, kws...)
end
function _QobjBuffers(::Type{T}, n, K, tspan, params, H_fwd, H_rev, G_fwd, G_rev,
                      kwargs) where T
    kws = (; kwargs...)
    alg = get(kws, :alg, Vern7(lazy=false)) # the default of `QuantumToolbox.sesolve`
    kws = Base.structdiff(kws, NamedTuple{(:alg,)})
    y0 = zeros(T, n)
    y0_aug = zeros(T, (K + 1) * n)
    int_fwd = _make_integrator(H_fwd, y0, tspan, params, alg, kws)
    int_rev = _make_integrator(H_rev, y0, tspan, params, alg, kws)
    int_gfwd = _make_integrator(G_fwd, y0_aug, tspan, params, alg, kws)
    int_grev = _make_integrator(G_rev, y0_aug, tspan, params, alg, kws)
    U0 = Matrix{T}(I, n, n)
    Y0 = zeros(T, (K + 1) * n, n)
    Y0[1:n, :] .= U0
    int_full = _make_integrator(H_fwd, U0, tspan, params, alg, kws)
    int_gfull = _make_integrator(G_fwd, Y0, tspan, params, alg, kws)
    empty = Matrix{T}(undef, 0, 0)
    return _QobjBuffers{T,typeof(int_fwd),typeof(int_rev),typeof(int_gfwd),typeof(int_grev),
                        typeof(int_full),typeof(int_gfull)}(
        y0, y0_aug, int_fwd, int_rev, int_gfwd, int_grev, U0, Y0, int_full, int_gfull,
        0, empty, Matrix{T}[], 0, empty, empty, Matrix{T}[])
end
function _right_buffers!(buf::_QobjBuffers{T}, n, K, k) where T
    if buf.k != k
        buf.UR = Matrix{T}(undef, n, k)
        buf.S = [Matrix{T}(undef, n, k) for _ in 1:K]
        buf.k = k
    end
    return buf.UR, buf.S
end
function _left_buffers!(buf::_QobjBuffers{T}, n, K, m) where T
    if buf.m != m
        buf.Lt = Matrix{T}(undef, n, m)
        buf.Y = Matrix{T}(undef, n, m)
        buf.DY = [Matrix{T}(undef, n, m) for _ in 1:K]
        buf.m = m
    end
    return buf.Lt, buf.Y, buf.DY
end

##### Construction

function QobjEvoStep{OP}(H::AbstractQuantumObject; nparams::Integer, tspan,
                         param_map=identity, kwargs...) where OP<:AbstractMatrix
    T = eltype(OP)
    if !(T <: Complex)
        throw(ArgumentError("`OP` must have a complex element type, got $OP"))
    end
    PT = real(T)
    K = Int(nparams)
    if K < 0
        throw(ArgumentError("`nparams` must be non-negative, got $nparams"))
    end
    if isoper(H)
        # U' = -im * H * U
        factor = T(-im)
    elseif issuper(H)
        # U' = L * U
        factor = one(T)
    else
        throw(ArgumentError("Unsupported quantum object type $(H.type)"))
    end
    Hevo = H isa QuantumObjectEvolution ? H : QobjEvo(H)
    n = size(Hevo, 1)

    consts = Any[]
    tds = Any[]
    _collect_terms!(consts, tds, Hevo.data, ())
    t0, t1 = tspan
    H_fwd, H_rev, G_fwd, G_rev = _build_operators(T, PT, n, K, consts, tds, factor, param_map,
                                                  t0 + t1)
    kws = (; kwargs...)
    params = zero(SVector{K,PT})
    buf = _QobjBuffers(T, n, K, (PT(t0), PT(t1)), params, H_fwd, H_rev, G_fwd, G_rev, kws)
    return QobjEvoStep{OP,K,T,PT,typeof(H_fwd),typeof(H_rev),typeof(G_fwd),typeof(G_rev),
                       typeof(kws),typeof(buf)}(
        H_fwd, H_rev, G_fwd, G_rev, n, PT(t0), PT(t1), kws, buf, params,
        Matrix{T}(undef, n, n), Vector{T}(undef, (K + 1) * n^2), false, false)
end

QobjEvoStep(H::AbstractQuantumObject; kwargs...) =
    QobjEvoStep{Matrix{complex(eltype(H))}}(H; kwargs...)

##### Evaluation

# Integrate from the initial state `y0` (with the current parameters) to `t1`, returning
# the final state (aliasing the integrator)
function _solve!(integ, y0, params)
    integ.p = params
    # Keep the algorithm caches (stage buffers), which would otherwise be reallocated, and
    # start from the last step size instead of re-estimating it (which allocates several
    # state-sized temporaries); the adaptive step control adjusts it as needed
    reinit!(integ, y0; reinit_cache=false, reset_dt=false)
    solve!(integ)
    return integ.u
end

@inline _block(y, n, k) = reshape(view(y, k * n^2 + 1:(k + 1) * n^2), n, n)

# Propagate the columns of `X` (n × k or a vector) as states with the integrator `ip`
# (n × n generator) or, with gradients (`S !== nothing`, an array of `K` outputs of the
# shape of `out`), with the integrator `ig` of the sensitivity system. For the reversed
# generator, this gives Uᵀ X and ∂(Uᵀ) X.
function _propagate_columns!(out, S, step::QobjEvoStep{OP,K,T}, ip, ig, X) where {OP,K,T}
    n = step.n
    buf = step.buf
    params = step.params
    if S === nothing
        y0 = buf.y0
        for j in 1:size(X, 2)
            copyto!(y0, view(X, :, j))
            y = _solve!(ip, y0, params)
            copyto!(view(out, :, j), y)
        end
    else
        y0 = buf.y0_aug
        fill!(view(y0, n + 1:(K + 1) * n), zero(T))
        for j in 1:size(X, 2)
            copyto!(view(y0, 1:n), view(X, :, j))
            y = _solve!(ig, y0, params)
            copyto!(view(out, :, j), view(y, 1:n))
            @inbounds for k in 1:K
                copyto!(view(S[k], :, j), view(y, k * n + 1:(k + 1) * n))
            end
        end
    end
    return out
end

# U * R (and ∂ₖU * R into `S[k]`) by propagating the columns of `R`
_right!(out, S, step::QobjEvoStep, R) =
    _propagate_columns!(out, S, step, step.buf.int_fwd, step.buf.int_gfwd, R)

# L * U (and L * ∂ₖU into `D[k]`) from (Uᵀ Lᵀ)ᵀ by propagating the rows of `L` with the
# reversed (transposed) generator
function _left!(out, D, step::QobjEvoStep{OP,K,T}, L) where {OP,K,T}
    n = step.n
    m = size(L, 1)
    Lt, Y, DY = _left_buffers!(step.buf, n, K, m)
    for a in 1:m
        Lt[:, a] .= view(L, a, :)
    end
    _propagate_columns!(Y, D === nothing ? nothing : DY, step, step.buf.int_rev,
                        step.buf.int_grev, Lt)
    transpose!(out, Y)
    if D !== nothing
        @inbounds for k in 1:K
            transpose!(D[k], DY[k])
        end
    end
    return out
end

# Full propagator and gradients (cached for the current parameters), with matrix states
# (one sparse × dense product per step, more efficient than propagating the columns)
function _propagator!(step::QobjEvoStep{OP,K,T}) where {OP,K,T}
    if !step.U_valid
        if step.Y_valid
            copyto!(step.U_cache, _block(step.Y_cache, step.n, 0))
        else
            copyto!(step.U_cache, _solve!(step.buf.int_full, step.buf.U0, step.params))
        end
        step.U_valid = true
    end
    return step.U_cache
end
function _sensitivities!(step::QobjEvoStep{OP,K,T}) where {OP,K,T}
    if !step.Y_valid
        n = step.n
        Y = step.Y_cache
        # Y = [U; ∂₁U; ...; ∂ₖU] ((K+1)n × n), stored as [vec(U); vec(∂₁U); ...]
        Ym = _solve!(step.buf.int_gfull, step.buf.Y0, step.params)
        @inbounds for k in 0:K
            copyto!(_block(Y, n, k), view(Ym, k * n + 1:(k + 1) * n, :))
        end
        step.Y_valid = true
        copyto!(step.U_cache, _block(Y, n, 0))
        step.U_valid = true
    end
    return step.Y_cache
end

# Result types (the types of the products with the operator, see `compute`) and the
# conversion of the computed dense arrays to them
_right_type(::Type{OP}, R) where OP = Base.promote_op(*, OP, typeof(R))
_left_type(::Type{OP}, L) where OP = Base.promote_op(*, typeof(L), OP)
_as_type(::Type{RT}, x) where RT = convert(RT, x)
_as_type(::Type{RT}, x::AbstractMatrix) where {RT<:Adjoint{<:Any,<:AbstractVector}} =
    adjoint(convert(fieldtype(RT, :parent), conj.(vec(x))))
_as_type(::Type{RT}, x::AbstractMatrix) where {RT<:Transpose{<:Any,<:AbstractVector}} =
    transpose(convert(fieldtype(RT, :parent), vec(x)))

# New dense arrays for U * R (results of the allocating forms)
_right_buffer(::Type{T}, n, R::AbstractVector) where T = Vector{T}(undef, n)
_right_buffer(::Type{T}, n, R) where T = Matrix{T}(undef, n, size(R, 2))
# The cached n × k buffers viewed with the shape of U * R (a vector for a vector `R`)
@inline _shaped(R::AbstractVector, X) = vec(X)
@inline _shaped(R, X) = X

function TimeSequence.compute(step::QobjEvoStep{OP,NParams,T}, grad, L, R) where {OP,NParams,T}
    n = step.n
    has_grad = !isempty(grad)
    has_grad && @assert length(grad) == NParams
    if R !== nothing
        if L === nothing
            # The results are returned directly (new arrays)
            UR = _right_buffer(T, n, R)
            S = has_grad ? [_right_buffer(T, n, R) for _ in 1:NParams] : nothing
            _right!(UR, S, step, R)
            RT = _right_type(OP, R)
            if has_grad
                @inbounds for k in 1:NParams
                    grad[k] = _as_type(RT, S[k])
                end
            end
            return _as_type(RT, UR)
        end
        UR, S = _right_buffers!(step.buf, n, NParams, size(R, 2))
        _right!(UR, has_grad ? S : nothing, step, R)
        if has_grad
            @inbounds for k in 1:NParams
                grad[k] = L * _shaped(R, S[k])
            end
        end
        return L * _shaped(R, UR)
    elseif L !== nothing
        LU = Matrix{T}(undef, size(L, 1), n)
        D = has_grad ? [Matrix{T}(undef, size(L, 1), n) for _ in 1:NParams] : nothing
        _left!(LU, D, step, L)
        LT = _left_type(OP, L)
        if has_grad
            @inbounds for k in 1:NParams
                grad[k] = _as_type(LT, D[k])
            end
        end
        return _as_type(LT, LU)
    end
    # Full operator (new objects, never aliasing the caches)
    if !has_grad
        return OP(_propagator!(step))
    end
    Y = _sensitivities!(step)
    @inbounds for k in 1:NParams
        grad[k] = OP(_block(Y, n, k))
    end
    return OP(_block(Y, n, 0))
end
TimeSequence.compute(step::QobjEvoStep, grad) = TimeSequence.compute(step, grad, nothing, nothing)

function TimeSequence.compute!(res, step::QobjEvoStep{OP,NParams,T}, grad, L, R) where {OP,NParams,T}
    n = step.n
    has_grad = !isempty(grad)
    has_grad && @assert length(grad) == NParams
    if R !== nothing
        if L === nothing
            _right!(res, has_grad ? grad : nothing, step, R)
            return res
        end
        UR, S = _right_buffers!(step.buf, n, NParams, size(R, 2))
        _right!(UR, has_grad ? S : nothing, step, R)
        if has_grad
            @inbounds for k in 1:NParams
                mul!(grad[k], L, _shaped(R, S[k]))
            end
        end
        mul!(res, L, _shaped(R, UR))
        return res
    elseif L !== nothing
        _left!(res, has_grad ? grad : nothing, step, L)
        return res
    end
    if !has_grad
        copyto!(res, _propagator!(step))
        return res
    end
    Y = _sensitivities!(step)
    @inbounds for k in 1:NParams
        copyto!(grad[k], _block(Y, n, k))
    end
    copyto!(res, _block(Y, n, 0))
    return res
end
function TimeSequence.compute!(res::OP, step::QobjEvoStep{OP}, grad) where OP
    if !ismutabletype(OP)
        throw(ArgumentError("In-place compute is not supported for immutable matrix type $OP"))
    end
    return TimeSequence.compute!(res, step, grad, nothing, nothing)
end

# U * R with the gradients L * ∂ₖU * R, propagating only the columns of `R`
function TimeSequence.compute_right(step::QobjEvoStep{OP,NParams,T}, grad, L, R, gradR) where {OP,NParams,T}
    n = step.n
    @assert length(grad) == NParams
    UR = _right_buffer(T, n, R)
    _, S = _right_buffers!(step.buf, n, NParams, size(R, 2))
    _right!(UR, S, step, R)
    @inbounds for k in 1:NParams
        grad[k] = L * _shaped(R, S[k])
    end
    return _as_type(_right_type(OP, R), UR)
end
function TimeSequence.compute_right!(res, step::QobjEvoStep{OP,NParams}, grad, L, R, gradR) where {OP,NParams}
    @assert length(grad) == NParams
    S = @view gradR[1:NParams]
    _right!(res, S, step, R)
    @inbounds for k in 1:NParams
        mul!(grad[k], L, S[k])
    end
    return res
end

end
