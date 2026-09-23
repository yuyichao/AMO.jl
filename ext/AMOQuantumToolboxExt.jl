#

module AMOQuantumToolboxExt

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff

import AMO.TimeSequence: QobjEvoStep
using AMO.TimeSequence: TimeSequence, _to_op
using QuantumToolbox: AbstractQuantumObject, QuantumObjectEvolution, QobjEvo, Qobj,
    isoper, issuper, sesolve, qeye
using SciMLOperators: AbstractSciMLOperator, AbstractSciMLScalarOperator, MatrixOperator,
    IdentityOperator, ScaledOperator, AddedOperator, update_coefficients, isconstant

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
# The propagator uses a `MatrixOperator` with the pattern `P` whose values are updated
# in-place. The sensitivity system
#     d/dt [vec(U); vec(∂₁U); ...; vec(∂ₖU)]
# uses the block lower triangular generator
#     I ⊗ I ⊗ A + Σₖ |k⟩⟨0| ⊗ I ⊗ ∂ₖA
# stored as a single sparse matrix with fixed structure, into which the values of
# `A` and `∂ₖA` are scattered with precomputed index maps.

struct _Terms{T,M,Λ<:NTuple{M,Tuple},PM}
    λss::Λ                 # scalar operators of the time-dependent terms
    param_map::PM
    H0nz::Vector{T}        # values of the constant term on `P`
    Onz::NTuple{M,Vector{T}} # values of the time-dependent terms on `P`
end

# In-place update of the (n × n) propagator generator
struct _PropUpdate{TT<:_Terms}
    terms::TT
end
function (f::_PropUpdate)(A, u, p, t)
    terms = f.terms
    pm = terms.param_map(p)
    nz = nonzeros(A)
    copyto!(nz, terms.H0nz)
    foreach(terms.λss, terms.Onz) do λs, Onz
        c = _scalars_value(λs, pm, t)
        @. nz += c * Onz
    end
    return A
end

# In-place update of the augmented generator
struct _AugUpdate{TT<:_Terms,T,K}
    terms::TT
    Anz::Vector{T}
    Bnz::NTuple{K,Vector{T}}
    destA::Vector{Int}
    srcA::Vector{Int}
    destB::NTuple{K,Vector{Int}}
    srcB::NTuple{K,Vector{Int}}
end
function (f::_AugUpdate{TT,T,K})(M, u, p, t) where {TT,T,K}
    terms = f.terms
    pm = terms.param_map(_dual_params(p))
    Anz = f.Anz
    Bnz = f.Bnz
    copyto!(Anz, terms.H0nz)
    foreach(b->fill!(b, zero(T)), Bnz)
    foreach(terms.λss, terms.Onz) do λs, Onz
        c = _scalars_value(λs, pm, t)
        cv = _dual_value(c)
        @. Anz += cv * Onz
        for k in 1:K
            ck = _dual_partial(c, k)
            if !iszero(ck)
                Bk = Bnz[k]
                @. Bk += ck * Onz
            end
        end
    end
    _scatter!(M, f, Anz, Bnz)
    return M
end
function _scatter!(M, f::_AugUpdate{TT,T,K}, Anz, Bnz) where {TT,T,K}
    nz = nonzeros(M)
    @inbounds for (q, s) in zip(f.destA, f.srcA)
        nz[q] = Anz[s]
    end
    for k in 1:K
        Bk = Bnz[k]
        @inbounds for (q, s) in zip(f.destB[k], f.srcB[k])
            nz[q] = Bk[s]
        end
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

function _build_operators(::Type{T}, ::Type{PT}, n, K, consts, tds, factor, param_map) where {T,PT}
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
    terms = _Terms(λss, param_map, H0nz, Onz)

    # Propagator generator
    Aprop = _with_values(P, copy(H0nz))
    H_prop = isempty(tds) ? MatrixOperator(Aprop) :
        MatrixOperator(Aprop; update_func! = _PropUpdate(terms))

    # Augmented generator
    n2 = n^2
    In = sparse(I, n, n)
    Mpat = kron(sparse(I, K + 1, K + 1), kron(In, P))
    for k in 1:K
        Ek = sparse([k + 1], [1], [true], K + 1, K + 1)
        Mpat = Mpat .| kron(Ek, kron(In, P))
    end
    destA = Int[]
    srcA = Int[]
    destB = [Int[] for _ in 1:K]
    srcB = [Int[] for _ in 1:K]
    rvM = rowvals(Mpat)
    for c in 1:size(Mpat, 2), q in nzrange(Mpat, c)
        r = rvM[q]
        br, rr = divrem(r - 1, n2)
        bc, cc = divrem(c - 1, n2)
        ii, a = divrem(rr, n)
        jj, b = divrem(cc, n)
        @assert ii == jj
        s = Pidx[a + 1, b + 1]
        @assert s > 0
        if br == bc
            push!(destA, q)
            push!(srcA, s)
        else
            @assert bc == 0
            push!(destB[br], q)
            push!(srcB[br], s)
        end
    end
    M = _with_values(Mpat, zeros(T, nnz(Mpat)))
    aug = _AugUpdate(terms, zeros(T, nnzP), Tuple(zeros(T, nnzP) for _ in 1:K),
                     destA, srcA, Tuple(destB), Tuple(srcB))
    # Initialize with the constant part
    _scatter!(M, aug, H0nz, aug.Bnz)
    H_aug = isempty(tds) ? MatrixOperator(M) : MatrixOperator(M; update_func! = aug)
    return QobjEvo(H_prop), QobjEvo(H_aug)
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
        factor = one(T)
    elseif issuper(H)
        # U' = L * U = -im * (im * L) * U
        factor = T(im)
    else
        throw(ArgumentError("Unsupported quantum object type $(H.type)"))
    end
    Hevo = H isa QuantumObjectEvolution ? H : QobjEvo(H)
    n = size(Hevo, 1)

    consts = Any[]
    tds = Any[]
    _collect_terms!(consts, tds, Hevo.data, ())
    H_prop, H_aug = _build_operators(T, PT, n, K, consts, tds, factor, param_map)

    ψ0_prop = qeye(n)
    ψ0 = zeros(T, (K + 1) * n^2)
    ψ0[1:n^2] .= vec(Matrix{T}(I, n, n))
    ψ0_aug = Qobj(ψ0)

    t0, t1 = tspan
    kws = (; kwargs...)
    return QobjEvoStep{OP,K,PT,typeof(H_prop),typeof(ψ0_prop),typeof(H_aug),typeof(ψ0_aug),
                       typeof(kws)}(H_prop, ψ0_prop, H_aug, ψ0_aug, n, PT(t0), PT(t1), kws,
                                    zero(SVector{K,PT}))
end

QobjEvoStep(H::AbstractQuantumObject; kwargs...) =
    QobjEvoStep{Matrix{complex(eltype(H))}}(H; kwargs...)

##### Evaluation

_solve(step::QobjEvoStep, H, ψ0) =
    sesolve(H, ψ0, [step.t0, step.t1]; params=step.params, progress_bar=Val(false),
            step.kwargs...).states[end].data

@inline _block(y, n, k) = reshape(view(y, k * n^2 + 1:(k + 1) * n^2), n, n)

function TimeSequence.compute(step::QobjEvoStep{OP,NParams}, grad) where {OP,NParams}
    n = step.n
    if isempty(grad)
        return _to_op(OP, _solve(step, step.H_prop, step.ψ0_prop))
    end
    @assert length(grad) == NParams
    y = _solve(step, step.H_aug, step.ψ0_aug)
    for k in 1:NParams
        grad[k] = _to_op(OP, _block(y, n, k))
    end
    return _to_op(OP, _block(y, n, 0))
end

function TimeSequence.compute!(res::OP, step::QobjEvoStep{OP,NParams}, grad) where {OP,NParams}
    if !ismutabletype(OP)
        throw(ArgumentError("In-place compute is not supported for immutable matrix type $OP"))
    end
    n = step.n
    if isempty(grad)
        copyto!(res, _solve(step, step.H_prop, step.ψ0_prop))
        return res
    end
    @assert length(grad) == NParams
    y = _solve(step, step.H_aug, step.ψ0_aug)
    for k in 1:NParams
        copyto!(grad[k], _block(y, n, k))
    end
    copyto!(res, _block(y, n, 0))
    return res
end

end
