##############################
# Load packages and functions:
##############################
using Pkg;
Pkg.activate(".");
Pkg.instantiate();
Pkg.status()
Pkg.add("StatsPlots")
using Distributions, Random, StatsBase, DataFrames, LinearAlgebra, Flux, ProgressMeter, StatsPlots;

include("simulate_communicatingCells.jl")
include("data_processing.jl")
include("dimension_reduction.jl")
gr(dpi=300)

################################################
# Initialize random seeds:
################################################
simulation_seeds = [3, 7, 13, 19, 29, 37, 43, 53, 61, 71]
vae_seeds = [5, 11, 17, 23, 31, 41, 47, 59, 67, 73]
################################################
# Set robustness testing objectives:
################################################
rob_test = "simulation_dim"    # "simulation", "simulation_dim", "vae"
################################################


if rob_test == "simulation"
    # Initialize a dictionary to store matching coefficients for each noise level
    matching_coefficients_per_noise = Dict{Float32, Vector{Float64}}()
    bp=boxplot()
    for noise in [0.3f0, 0.5f0, 0.7f0, 0.75f0]
        matching_coefficients = Float64[]  # Initialize an empty array for this noise level
        println("Noise: ", noise)
        for seed in simulation_seeds
            println("Seed: ", seed)
            # Generate the simulated data (set a seed for reproducibility):
            communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
            dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=seed, communication_graph=communication_graph)
            # Preprocess the data:
            X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=noise, save_figures=false)
            B, Ŷ, communication_idxs, Y_t, matching_coefficient = iterative_rematching(5, X, Y, groups=[1:250, 251:500, 501:750, 751:1000], save_figures=false)
            push!(matching_coefficients, matching_coefficient)
        end
        boxplot!([string(Int(round(noise * 100)))* "%"], matching_coefficients, whisker_range=Inf, whisker_width=0.3, legend=false, color="blue", fillalpha=0.5, ylabel="Correct Assignments (relative)", xlabel="Initial Noise Percentage", title="Stability: Different Data Seeds", ylim=(0, 1))
    end
    display(bp)
    savefig("output/auto_output/robustness_simulation.png")
end

if rob_test == "vae"
    # Initialize a dictionary to store matching coefficients for each noise level
    matching_coefficients_per_noise = Dict{Float32, Vector{Float64}}()
    bp=boxplot()
    for noise in [0.3f0, 0.5f0, 0.7f0, 0.75f0]
        println("Noise: ", noise)
        matching_coefficients = Float64[]  # Initialize an empty array for this noise level
        communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
        dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=3, communication_graph=communication_graph)
        # Preprocess the data:
        X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=noise, save_figures=false)
        for seed in vae_seeds
            println("Seed: ", seed)
            Z_X, Z_Y = dimension_reduction(X, Y, "VAE", 4, save_figures=false, seed=seed)
            B, Ŷ, communication_idxs, Y_t, matching_coefficient = iterative_rematching(5, X, copy(Z_Y'), Z=copy(Z_X'), groups=[1:250, 251:500, 501:750, 751:1000], save_figures=false)
            push!(matching_coefficients, matching_coefficient)
        end
        boxplot!([string(Int(round(noise * 100)))* "%"], matching_coefficients, whisker_range=Inf, whisker_width=0.3, legend=false, color="blue", fillalpha=0.5, ylabel="Correct Assignments (relative)", xlabel="Initial Noise Percentage", title="Stability: Different VAE Model Seeds", ylim=(0, 1))
    end
    display(bp)
    savefig("output/auto_output/robustness_vae.png")
end

if rob_test == "simulation_dim"
    # Initialize a dictionary to store matching coefficients for each noise level
    matching_coefficients_per_noise = Dict{Float32, Vector{Float64}}()
    bp=boxplot()
    for noise in [0.3f0, 0.5f0, 0.7f0, 0.75f0]
        println("Noise: ", noise)
        matching_coefficients = Float64[]  # Initialize an empty array for this noise level
        for seed in simulation_seeds
            println("Seed: ", seed)
            # Generate the simulated data (set a seed for reproducibility):
            communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
            dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=seed, communication_graph=communication_graph)
            # Preprocess the data:
            X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=noise, save_figures=false)
            Z_X, Z_Y = dimension_reduction(X, Y, "VAE", 4, save_figures=false, seed=5)
            B, Ŷ, communication_idxs, Y_t, matching_coefficient = iterative_rematching(5, X, copy(Z_Y'), Z=copy(Z_X'), groups=[1:250, 251:500, 501:750, 751:1000], save_figures=false)
            push!(matching_coefficients, matching_coefficient)
        end
        boxplot!([string(Int(round(noise * 100)))* "%"], matching_coefficients, whisker_range=Inf, whisker_width=0.3, legend=false, color="blue", fillalpha=0.5, ylabel="Correct Assignments (relative)", xlabel="Initial Noise Percentage", title="Stability: Different Data Seeds (with VAE)", ylim=(0, 1))
    end
    display(bp)
    savefig("output/auto_output/robustness_simulation_dim.png")
end