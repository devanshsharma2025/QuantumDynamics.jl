module QCHEOM

using OrdinaryDiffEq
using ..HEOMStructure
using ..SpectralDensities, ..Solvents, ..Utilities

function propagate(; Hamiltonian::Matrix{ComplexF64}, Jw::AbstractVector{<:SpectralDensities.SpectralDensity}, solvent::Solvents.Solvent, ρ0::Matrix{ComplexF64}, β::Real, dt::Real, ntimes::Int, Lmax::Int, sops::Vector{Matrix{ComplexF64}}, verbose::Bool=false)
    num_modes = 0
    nbaths = length(Jw)
    γ = zeros(nbaths, num_modes + 1)
    c = zeros(ComplexF64, nbaths, num_modes + 1)
    for (i, jw) in enumerate(Jw)
        @assert typeof(jw) == SpectralDensities.DrudeLorentz "HEOM has only been implemented for the Drude-Lorentz spectral density."
        γj, cj = SpectralDensities.matsubara_decomposition_imaginary(jw, num_modes, β)
        @inbounds γ[i, :] .= γj
        @inbounds c[i, :] .= cj
        @info "Decomposed bath number $i."
    end
    nveclist, npluslocs, nminuslocs = HEOMStructure.setup_simulation(length(Jw), num_modes, Lmax)
    @info "Setup complete. Starting run"
    @info "Number of ADOs used: $(length(nveclist))"

    decay = zeros(Float64, length(nveclist))
    for (i, nvec) in enumerate(nveclist)
        decay[i] = sum(nvec .* γ)
    end

    tspan = (0.0, ntimes * dt)
    sdim = size(ρ0, 1)
    Nh = length(nveclist)
    ρs = zeros(ComplexF64, ntimes+1, sdim, sdim)
    workspace = zeros(ComplexF64, sdim, sdim)
    tmp1 = zeros(ComplexF64, sdim, sdim)
    ρ0exp = zeros(ComplexF64, sdim, sdim, Nh)
    ρ0exp[:, :, 1] .= ρ0
    for (j, ps) in enumerate(solvent)
        verbose && @info "$(j)th iteration"
        params = HEOMStructure.HEOMParams(Hamiltonian, nothing, nothing, (solvent, ps), sops, nveclist, npluslocs, nminuslocs, γ, c, 0.0, β, decay, workspace, tmp1)
        prob = ODEProblem{true}(HEOMStructure.scaled_HEOM_RHS!, ρ0exp, tspan, params)
        sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5, saveat=dt)
        for t=1:length(sol)
            @inbounds ρs[t, :, :] .+= sol.u[t][:, :, 1]
        end
    end
    ρs /= solvent.nsamples
    0:dt:ntimes*dt, ρs
end

end
