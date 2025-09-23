#

module Trap

const c = 299792458
const h = 6.62607015e-34
const ħ = h / 2π

@generated function _check_one_kwarg(kws)
    names = fieldnames(kws)
    found_idx = 0
    for (i, type) in enumerate(fieldtypes(kws))
        if type === Nothing
            continue
        end
        if found_idx == 0
            found_idx = i
            continue
        end
        msg = "Argument conflict: only one of $(join(names, ", ")) may be provided"
        return :(throw(ArgumentError($msg)))
    end
    if found_idx == 0
        msg = "Argument missing: one of $(join(names, ", ")) must be provided"
        return :(throw(ArgumentError($msg)))
    end
    return QuoteNode(names[found_idx])
end

@inline function η(m; fm=nothing, ωm=nothing, kp=nothing, λp=nothing, νp=nothing)
    name = _check_one_kwarg((;fm, ωm))
    if name === :fm
        ωm = float(typeof(fm))(2π * fm)
    else
        @assert name === :ωm
    end
    name = _check_one_kwarg((;λp, kp, νp))
    if name === :λp
        kp = float(typeof(λp))(2π / λp)
    elseif name === :νp
        kp = float(typeof(νp))(2π / c * νp)
    else
        @assert name === :kp
    end
    T = float(promote_type(typeof(m), typeof(ωm), typeof(kp)))
    return sqrt(T(ħ) / 2 / m / ωm) * kp
end


end
