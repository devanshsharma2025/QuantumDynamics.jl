module QCHEOM

using OrdinaryDiffEq
using ..HEOMStructure
using ..SpectralDensities, ..Solvents, ..Utilities

function single_propagate(phasespacepoints, solvent, Hamiltonian, sops, nveclist, npluslocs, nminuslocs, γ, c, β, decay, ρ0exp, ntimes, dt, verbose)
    tspan = (0.0, ntimes * dt)
    sdim = size(Hamiltonian, 1)
    ρs = zeros(ComplexF64, ntimes+1, sdim, sdim)
    workspace = zeros(ComplexF64, sdim, sdim)
    tmp1 = zeros(ComplexF64, sdim, sdim)
    Npoints = length(phasespacepoints)
    update_len = max(1, Npoints ÷ 10)
    for (j, ps) in enumerate(phasespacepoints)
        params = HEOMStructure.HEOMParams(Hamiltonian, nothing, nothing, (solvent, ps), sops, nveclist, npluslocs, nminuslocs, γ, c, 0.0, β, decay, workspace, tmp1)
        prob = ODEProblem{true}(HEOMStructure.scaled_HEOM_RHS!, ρ0exp, tspan, params)
        sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5, saveat=dt)
        for t=1:length(sol)
            @inbounds ρs[t, :, :] .+= sol.u[t][:, :, 1]
        end
        if verbose && (j % update_len == 0)
            @info "Initial condition number $j of $(Npoints) done."
        end
    end
    ρs, Npoints
end

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

    sdim = size(ρ0, 1)
    Nh = length(nveclist)
    ρ0exp = zeros(ComplexF64, sdim, sdim, Nh)
    ρ0exp[:, :, 1] .= ρ0
    chunks = Iterators.partition(solvent, cld(length(solvent), Threads.nthreads()))
    ρtasks = map(enumerate(chunks)) do (ind, chunk)
        Threads.@spawn single_propagate(chunk, solvent, Hamiltonian, sops, nveclist, npluslocs, nminuslocs, γ, c, β, decay, copy(ρ0exp), ntimes, dt, verbose && (ind==1))
    end
    results = fetch.(ρtasks)
    ρs = zero(results[1][1])
    nsamples = 0
    for (res, j) in results
        ρs .+= res
        nsamples += j
    end

    0:dt:ntimes*dt, ρs/nsamples
end

end
