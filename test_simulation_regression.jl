using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

using Distributions, Plots, Random, StatsBase, DataFrames, LinearAlgebra, Flux, ProgressMeter;

include("simulate_communicatingCells.jl")
include("data_processing.jl")
include("AE.jl")
gr(dpi=300)

################################################
# Initialize for (possible) dimension reduction:
################################################
dim_reduction = "AE"    # "none", "AE", "VAE"
AE_architecture = "deep"   # "simple", "deep"
latent_dim = 4
################################################

# Generate the simulated data (set a seed for reproducibility):
communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];
dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=7, communication_graph = communication_graph)

# Preprocess the data:
X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=0.3f0, save_figures=true)

if dim_reduction in ["AE", "VAE"]
    Z_X, Z_Y = dimension_reduction(X, Y, dim_reduction, AE_architecture, latent_dim, save_figures=true)
    B, Ŷ, communication_idxs, Y_t = iterative_rematching(10, X, copy(Z_Y'), Z=copy(Z_X'), save_figures=true)
else
    B, Ŷ, communication_idxs, Y_t = iterative_rematching(10, X, Y, save_figures=true)
end