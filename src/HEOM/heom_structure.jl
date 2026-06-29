module HEOMStructure

using ..SpectralDensities, ..Solvents, ..Utilities

"""
    get_vecs(len::Int, L::Int)

Get a vector of vectors of length `len`, where the sum is L.
"""
function get_vecs(len::Int, L::Int)
    len == 1 && return [[L]]
    ans = Vector{Vector{typeof(L)}}()
    for j = 0:L
        rest = get_vecs(len - 1, L - j)
        curr = [cat(j, r; dims=1) for r in rest]
        append!(ans, curr)
    end
    ans
end

"""
    setup_simulation(num_baths::Int, num_modes::Int, Lmax::Int)

Sets up the simulation parameters for a problem with `num_baths` baths, `num_modes` extra matsubara modes, and a hierarchy `Lmax` levels deep.

Returns a tuple of:
- `nveclist`: List of the possible subscripts, `n`, in HEOM. Each element in the list is a represented as a matrix. Every row corresponds to a bath.
- `npluslocs[b,m,l]`: Given the `l`th nvector, returns the location of the nvector if the `b`th bath's `m`th Matsubara mode is increased by one.
- `nminuslocs[b,m,l]`: Given the `l`th nvector, returns the location of the nvector if the `b`th bath's `m`th Matsubara mode is decreased by one.
"""
function setup_simulation(num_baths::Int, num_modes::Int, Lmax::Int)
    nveclist = Vector{Matrix{Int}}()
    len = num_baths * (num_modes + 1)
    for L = 0:Lmax
        vecs = get_vecs(len, L)
        for v in vecs
            push!(nveclist, reshape(v, num_baths, num_modes + 1))
        end
    end
    Nh = length(nveclist)
    index = Dict{NTuple{len,Int}, Int}()
    for (i, v) in enumerate(nveclist)
        index[Tuple(vec(v))] = i
    end

    npluslocs  = zeros(Int, num_baths, num_modes + 1, Nh)
    nminuslocs = zeros(Int, num_baths, num_modes + 1, Nh)
    for (j, nvec) in enumerate(nveclist)
        base_key = Tuple(vec(nvec))
        for m = 1:num_baths
            for k = 1:(num_modes + 1)
                nvec_plus = copy(nvec)
                nvec_plus[m, k] += 1
                npluslocs[m, k, j] = get(index, Tuple(vec(nvec_plus)), 0)
                if nvec[m, k] > 0
                    nvec_minus = copy(nvec)
                    nvec_minus[m, k] -= 1
                    nminuslocs[m, k, j] = get(index, Tuple(vec(nvec_minus)), 0)
                else
                    nminuslocs[m, k, j] = 0
                end
            end
        end
    end

    nveclist, npluslocs, nminuslocs
end

struct HEOMParams{Ltype <: Union{Nothing, Vector{Matrix{ComplexF64}}}, EField <: Union{Nothing, Vector{Utilities.ExternalField}, Tuple{Solvents.Solvent, Solvents.PhaseSpace}}}
    H::Matrix{ComplexF64}
    L::Ltype
    LdagL::Ltype
    external_fields::EField
    coupl::Vector{Matrix{ComplexF64}}
    nveclist
    npluslocs
    nminuslocs
    γ
    c
    Δk
    β
    decay::Vector{Float64}
    workspace::Matrix{ComplexF64}
    tmp1::Matrix{ComplexF64}
end

function get_eff_hamiltonian!(tmp1, H, ext_fields::Nothing, t)
    tmp1 .= H
    nothing
end
function get_eff_hamiltonian!(tmp1, H, ext_fields::Vector{Utilities.ExternalField}, t)
    tmp1 .= H
    for ef in ext_fields
        tmp1 .+= ef.V(t) * ef.coupling_op
    end
    nothing
end
function get_eff_hamiltonian!(tmp1, H, ext_fields::Tuple{Solvents.Solvent, Solvents.PhaseSpace}, t)
    tmp1 .= H + Solvents.get_Vint(ext_fields[2], ext_fields[1], t)
    nothing
end
function get_base_eom!(dρ, ρ, H, tmp1, external_fields, t)
    get_eff_hamiltonian!(tmp1, H, external_fields, t)
    for n in axes(ρ, 3)
        dρ[:,:,n] .= -1im * Utilities.nh_commutator(tmp1, ρ[:,:,n])
    end
    nothing
end

function uncoupled_eom!(dρ, ρ, params::HEOMParams{Nothing, T}, t) where T
    get_base_eom!(dρ, ρ, params.H, params.tmp1, params.external_fields, t)
    nothing
end
function uncoupled_eom!(dρ, ρ, params::HEOMParams{Vector{Matrix{ComplexF64}}, T}, t) where T
    get_base_eom!(dρ, ρ, params.H, params.tmp1, params.external_fields, t)
    for n in axes(ρ, 3)
        for (L, LdagL) in zip(params.L, params.LdagL)
            dρ[:, :, n] .+= L * ρ[:, :, n] * L' .- 0.5 .* LdagL * ρ[:, :, n] .- 0.5 .* ρ[:, :, n] * LdagL
        end
    end
    nothing
end
function scaled_HEOM_RHS!(dρ, ρ, params, t)
    @inbounds begin
        uncoupled_eom!(dρ, ρ, params, t)
        for n in axes(ρ, 3)
            @. dρ[:, :, n] -= params.decay[n] * ρ[:, :, n]
            for (Δk, co) in zip(params.Δk, params.coupl)
                dρ[:, :, n] .-= Δk .* Utilities.commutator(co, Utilities.commutator(co, ρ[:, :, n]))
            end

            @views begin
                nvec = params.nveclist[n]
                npluslocs = params.npluslocs[:, :, n]
                nminuslocs = params.nminuslocs[:, :, n]
                ρplus = params.workspace
            end
            for (m, co) in enumerate(params.coupl)
                fill!(ρplus, 0.0)
                for k in axes(npluslocs, 2)
                    if npluslocs[m, k] > 0
                        ρplus .+= sqrt((nvec[m, k] + 1) * abs(params.c[m, k])) * ρ[:, :, npluslocs[m, k]]
                    end
                    if nminuslocs[m, k] > 0
                        dρ[:, :, n] .+= -1im * sqrt(nvec[m, k] / abs(params.c[m, k])) * (params.c[m, k] * co * ρ[:, :, nminuslocs[m, k]] .- conj(params.c[m, k]) * ρ[:, :, nminuslocs[m, k]] * co)
                    end
                end
                dρ[:, :, n] .+= -1im * Utilities.commutator(co, ρplus)
            end
        end
    end
    nothing
end
function unscaled_HEOM_RHS!(dρ, ρ, params, t)
    @inbounds begin
        uncoupled_eom!(dρ, ρ, params, t)
        for n in axes(ρ, 3)
            @. dρ[:, :, n] -= params.decay[n] * ρ[:, :, n]
            for (Δk, co) in zip(params.Δk, params.coupl)
                dρ[:, :, n] .-= Δk .* Utilities.commutator(co, Utilities.commutator(co, ρ[:, :, n]))
            end

            @views begin
                nvec = params.nveclist[n]
                npluslocs = params.npluslocs[:, :, n]
                nminuslocs = params.nminuslocs[:, :, n]
                ρplus = params.workspace
            end
            for (m, co) in enumerate(params.coupl)
                fill!(ρplus, 0.0)
                for k in axes(npluslocs, 2)
                    if npluslocs[m, k] > 0
                        ρplus .+= sqrt((nvec[m, k] + 1) * abs(params.c[m, k])) * ρ[:, :, npluslocs[m, k]]
                    end
                    if nminuslocs[m, k] > 0
                        dρ[:, :, n] .+= -1im * sqrt(nvec[m, k] / abs(params.c[m, k])) * (params.c[m, k] * co * ρ[:, :, nminuslocs[m, k]] .- conj(params.c[m, k]) * ρ[:, :, nminuslocs[m, k]] * co)
                    end
                end
                dρ[:, :, n] .+= -1im * Utilities.commutator(co, ρplus)
            end
        end
    end
    nothing
end

end
