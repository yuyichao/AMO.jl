#

module AMOQuantumToolboxExt

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff

import AMO.TimeSequence: QobjEvoStep
using AMO.TimeSequence: TimeSequence, _to_op, _lrmul, _lmul
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
# Four `MatrixOperator`s with in-place updates are built:
#   H_fwd: A(p, t)                          the generator (n × n)
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

# In-place update of the (n × n) generator
struct _PropUpdate{TT<:_Terms}
    terms::TT
end
function (f::_PropUpdate)(A, u, p, t)
    terms = f.terms
    pm = terms.param_map(p)
    te = _time(terms, t)
    nz = nonzeros(A)
    copyto!(nz, terms.H0nz)
    foreach(terms.λss, terms.Onz) do λs, Onz
        c = _scalars_value(λs, pm, te)
        @. nz = muladd(c, Onz, nz)
    end
    return A
end

# In-place update of the sensitivity system
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
    te = _time(terms, t)
    Anz = f.Anz
    Bnz = f.Bnz
    copyto!(Anz, terms.H0nz)
    foreach(b->fill!(b, zero(T)), Bnz)
    foreach(terms.λss, terms.Onz) do λs, Onz
        c = _scalars_value(λs, pm, te)
        cv = _dual_value(c)
        @. Anz = muladd(cv, Onz, Anz)
        for k in 1:K
            ck = _dual_partial(c, k)
            if !iszero(ck)
                Bk = @inbounds Bnz[k]
                @. Bk = muladd(ck, Onz, Bk)
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
    @inbounds for k in 1:K
        Bk = Bnz[k]
        for (q, s) in zip(f.destB[k], f.srcB[k])
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

function _make_prop(P, terms::_Terms, constant::Bool)
    A = _with_values(P, copy(terms.H0nz))
    return QobjEvo(constant ? MatrixOperator(A) : MatrixOperator(A; update_func! = _PropUpdate(terms)))
end

function _make_aug(Mpat, ::Type{T}, nnzP, K, terms::_Terms, constant::Bool, destA, srcA, destB, srcB) where T
    M = _with_values(Mpat, zeros(T, nnz(Mpat)))
    aug = _AugUpdate(terms, zeros(T, nnzP), Tuple(zeros(T, nnzP) for _ in 1:K),
                     destA, srcA, Tuple(destB), Tuple(srcB))
    # Initialize with the constant part
    _scatter!(M, aug, terms.H0nz, aug.Bnz)
    return QobjEvo(constant ? MatrixOperator(M) : MatrixOperator(M; update_func! = aug))
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
    destA = Int[]
    srcA = Int[]
    destB = [Int[] for _ in 1:K]
    srcB = [Int[] for _ in 1:K]
    rvM = rowvals(Mpat)
    for c in 1:size(Mpat, 2), q in nzrange(Mpat, c)
        r = rvM[q]
        br, rr = divrem(r - 1, n)
        bc, cc = divrem(c - 1, n)
        s = Pidx[rr + 1, cc + 1]
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

    H = _make_prop(P, terms, constant)
    G = _make_aug(Mpat, T, nnzP, K, terms, constant, destA, srcA, destB, srcB)
    return H, G
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
    t0, t1 = tspan
    H_fwd, H_rev, G_fwd, G_rev = _build_operators(T, PT, n, K, consts, tds, factor, param_map,
                                                  t0 + t1)
    ψ0_prop = qeye(n)
    kws = (; kwargs...)
    return QobjEvoStep{OP,K,T,PT,typeof(H_fwd),typeof(H_rev),typeof(G_fwd),typeof(G_rev),
                       typeof(ψ0_prop),typeof(kws)}(
        H_fwd, H_rev, G_fwd, G_rev, ψ0_prop, n, PT(t0), PT(t1), kws, zero(SVector{K,PT}),
        Matrix{T}(undef, n, n), Vector{T}(undef, (K + 1) * n^2), false, false)
end

QobjEvoStep(H::AbstractQuantumObject; kwargs...) =
    QobjEvoStep{Matrix{complex(eltype(H))}}(H; kwargs...)

##### Evaluation

_tlist(step::QobjEvoStep) = [step.t0, step.t1]
_solve(step::QobjEvoStep, H, ψ0) =
    sesolve(H, ψ0, _tlist(step); params=step.params, progress_bar=Val(false),
            step.kwargs...).states[end].data
_solve_ket(step::QobjEvoStep, H, y0::AbstractVector) = _solve(step, H, Qobj(y0))

@inline _block(y, n, k) = reshape(view(y, k * n^2 + 1:(k + 1) * n^2), n, n)

# Propagate the columns of `X` (n × k or a vector) as states with `H` (n × n) or, with
# gradients (`S !== nothing`, an array of `K` outputs of the shape of `out`), with the
# sensitivity system `G`. For the reversed generator, this gives U† X and ∂(U†) X.
function _propagate_columns!(out, S, step::QobjEvoStep{OP,K,T}, H, G, X) where {OP,K,T}
    n = step.n
    if S === nothing
        y0 = Vector{T}(undef, n)
        for j in 1:size(X, 2)
            copyto!(y0, view(X, :, j))
            y = _solve_ket(step, H, y0)
            copyto!(view(out, :, j), y)
        end
    else
        y0 = zeros(T, (K + 1) * n)
        for j in 1:size(X, 2)
            copyto!(view(y0, 1:n), view(X, :, j))
            y = _solve_ket(step, G, y0)
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
    _propagate_columns!(out, S, step, step.H_fwd, step.G_fwd, R)

# L * U (and L * ∂ₖU into `D[k]`) from (Uᵀ Lᵀ)ᵀ by propagating the rows of `L` with the
# reversed (transposed) generator
function _left!(out, D, step::QobjEvoStep{OP,K,T}, L) where {OP,K,T}
    n = step.n
    m = size(L, 1)
    Lt = Matrix{T}(undef, n, m)
    for a in 1:m
        Lt[:, a] .= view(L, a, :)
    end
    Y = Matrix{T}(undef, n, m)
    DY = D === nothing ? nothing : [Matrix{T}(undef, n, m) for _ in 1:K]
    _propagate_columns!(Y, DY, step, step.H_rev, step.G_rev, Lt)
    transpose!(out, Y)
    if D !== nothing
        @inbounds for k in 1:K
            transpose!(D[k], DY[k])
        end
    end
    return out
end

# Full propagator and gradients (cached for the current parameters)
function _propagator!(step::QobjEvoStep)
    if !step.U_valid
        if step.Y_valid
            copyto!(step.U_cache, _block(step.Y_cache, step.n, 0))
        else
            copyto!(step.U_cache, _solve(step, step.H_fwd, step.ψ0_prop))
        end
        step.U_valid = true
    end
    return step.U_cache
end
function _sensitivities!(step::QobjEvoStep{OP,K,T}) where {OP,K,T}
    if !step.Y_valid
        n = step.n
        Y = step.Y_cache
        U = _block(Y, n, 0)
        S = [_block(Y, n, k) for k in 1:K]
        _right!(U, S, step, Matrix{T}(I, n, n))
        step.Y_valid = true
        copyto!(step.U_cache, U)
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

# Dense buffer for U * R
_right_buffer(::Type{T}, n, R::AbstractVector) where T = Vector{T}(undef, n)
_right_buffer(::Type{T}, n, R) where T = Matrix{T}(undef, n, size(R, 2))

function TimeSequence.compute(step::QobjEvoStep{OP,NParams,T}, grad, L, R) where {OP,NParams,T}
    n = step.n
    has_grad = !isempty(grad)
    has_grad && @assert length(grad) == NParams
    if R !== nothing
        UR = _right_buffer(T, n, R)
        S = has_grad ? [_right_buffer(T, n, R) for _ in 1:NParams] : nothing
        _right!(UR, S, step, R)
        if L === nothing
            RT = _right_type(OP, R)
            if has_grad
                @inbounds for k in 1:NParams
                    grad[k] = _as_type(RT, S[k])
                end
            end
            return _as_type(RT, UR)
        end
        if has_grad
            @inbounds for k in 1:NParams
                grad[k] = L * S[k]
            end
        end
        return L * UR
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
        UR = _right_buffer(T, n, R)
        S = has_grad ? [_right_buffer(T, n, R) for _ in 1:NParams] : nothing
        _right!(UR, S, step, R)
        if has_grad
            @inbounds for k in 1:NParams
                mul!(grad[k], L, S[k])
            end
        end
        mul!(res, L, UR)
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
    S = [_right_buffer(T, n, R) for _ in 1:NParams]
    _right!(UR, S, step, R)
    @inbounds for k in 1:NParams
        grad[k] = L * S[k]
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
