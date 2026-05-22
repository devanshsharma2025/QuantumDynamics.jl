module Polaron

using ..SpectralDensities
using ..Utilities
using LinearAlgebra

function full_polaron_transform(; Hamiltonian::AbstractMatrix{<:Number}, Jw::AbstractVector{SpectralDensities.SpectralDensity}, svec=[1.0 -1.0], β::Real)
    H = copy(Hamiltonian)
    N = size(Hamiltonian, 1)
    Γ = zeros(N, N)
    for (i, J) in enumerate(Jw)
        λ = SpectralDensities.reorganization_energy(J)
        H .-= diagm(svec[i, :].^2 .* λ) 
	    polaron_factor = SpectralDensities.polaron_shielding(J, β)
        for j=1:N, k=j+1:N
	        Δs = svec[i,j] - svec[i,k]
            Γ[j, k] += Δs^2 * polaron_factor
            Γ[k, j] += Δs^2 * polaron_factor
        end
    end
    for j = 1:N, k = j+1:N
        H[j, k] *= exp(-1.0 / 2 * Γ[j, k])
        H[k, j] *= exp(-1.0 / 2 * Γ[k, j])
    end
    H
end

function matrix_exponential(mat::AbstractMatrix{<:Number})
    vals, vecs = eigen(mat)
    exp_mat_diag = diagm(exp.(vals))
    exp_mat = vecs * exp_mat_diag * (vecs)'

    exp_mat
end

function variational_site_energy(; J::SpectralDensities.SpectralDensity, svec=[1.0 -1.0], F::Vector{Float64})
    energy_correction = svec[i, :].^2 .* Utilities.trapezoid(J.ω, (J.jw ./ J.ω .* (F .^ 2 .- 2 * F))) / π

    energy_correction
end

function variational_polaron_factor(; J::SpectralDensities.SpectralDensity, β::Real, F::Vector{Float64})
    J_var = copy(J)
    J_var.jw .*= F .^ 2
    var_polaron_factor = SpectralDensities.polaron_shielding(J_var, β)

    var_polaron_factor
end

function variational_hamiltonian(; Hamiltonian::AbstractMatrix{<:Number}, Jw::AbstractVector{SpectralDensities.SpectralDensity}, svec=[1.0 -1.0], β::Real, var_param::Vector)
    H = copy(Hamiltonian)
    N = size(Hamiltonian, 1)
    Γ = zeros(N, N)
    for (i, J) in enumerate(Jw)
        energy_correction = variational_site_energy(; J=J, svec=svec, F=var_param[i])
        H .+= diagm(energy_correction) 
	    var_polaron_factor = variational_polaron_factor(; J=J, β=β, F=var_param[i])
        for j=1:N, k=j+1:N
	        Δs = svec[i,j] - svec[i,k]
            Γ[j, k] += Δs^2 * var_polaron_factor
            Γ[k, j] += Δs^2 * var_polaron_factor
        end
    end
    for j = 1:N, k = j+1:N
        H[j, k] *= exp(-1.0 / 2 * Γ[j, k])
        H[k, j] *= exp(-1.0 / 2 * Γ[k, j])
    end
    H, Γ
end

function variational_thermal_system_densitymatrix(; var_Hamiltonian::AbstractMatrix{<:Number}, β::Real)
    exp_βH = matrix_exponential(- β * var_Hamiltonian) 
    Z = tr(exp_βH)
    var_sys_ρeq = exp_βH / Z

    var_sys_ρeq
end

function variational_parameter(; Hamiltonian::AbstractMatrix{<:Number}, Jw::AbstractVector{SpectralDensities.SpectralDensity}, β::Real, svec=[1.0 -1.0], old_var_param::Vector)
    H = copy(Hamiltonian)
    N = size(Hamiltonian, 1)

    varH, Γ = variational_hamiltonian(; Hamiltonian=Hamiltonian, Jw=Jw, svec=svec, β=β, var_param=old_var_param)
    κ = matrix_exponential(-1/2 * Γ)

    ρSeq = variational_thermal_system_densitymatrix(; var_Hamiltonian=varH, β=β)

    var_param = Vector()
    for (i, J) in enumerate(Jw)
        ρsn_sum = 0.0 + 0.0im
        ρnm_hκΔsmn_sum = 0.0 + 0.0im
        for j=1:N
            ρsn_sum += ρSeq[j,j] * svec[j,j] ^ 2
            for k=1:N
                if k != j
	                Δs = svec[i,j] - svec[i,k]
                    ρnm_hκΔsmn_sum += ρSeq[k,j] * H[j,k] * κ[j,k] * (Δs ^ 2)
                end
            end
        end
        F = 1 ./ (1 .- (coth.(J.ω * β / 2) ./ J.ω .* ρnm_hκΔsmn_sum ./ ρsn_sum) / 2)
        push!(var_param, F)
    end

    var_param, varH
end

function initial_variational_param_constructor(; Jw::AbstractVector{SpectralDensities.SpectralDensity}, bathwise_init_param::Vector{Float64})
    @assert all(0.0 <= x <= 1.0 for x in bathwise_init_param)

    init_var_param = Vector()
    for (n, val) in enumerate(bathwise_init_param)
        push!(init_var_param, fill(val, length((Jw[n]).jw)))
    end

    init_var_param
end

function variational_polaron_transform(; Hamiltonian::AbstractMatrix{<:Number}, Jw::AbstractVector{SpectralDensities.SpectralDensity}, β::Real, svec=[1.0 -1.0], init_var_param::Vector, tolerance::Real)
    @assert length(init_var_param) == length(Jw)
    @assert all(length(init_var_param[i]) == length(Jw.jw[i]) for i in eachindex(init_var_param))
    for i in eachindex(init_var_param)
        @assert all(0.0 <= x <= 1.0 for x in init_var_param[i])
    end

    old_Fs = copy(init_var_param)
    while true
        converged = 1
        new_Fs, varH = variational_parameter(; Hamiltonian=Hamiltonian, Jw=Jw, β=β, svec=svec, old_var_param=old_Fs)
        for i=1:length(Jw)
            if max(abs.(new_Fs[i] .- old_Fs[i])) > tolerance
                converged *= 0
            end 
        end
        if converged == 1
            return varH
        else
            old_Fs = copy(new_Fs)
        end
    end
end


end