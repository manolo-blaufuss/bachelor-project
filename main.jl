##############################
# Load packages and functions:
##############################
using Pkg;
Pkg.activate(".");
Pkg.instantiate();
Pkg.status()
using Distributions, Plots, Random, StatsBase, DataFrames, LinearAlgebra, Flux, ProgressMeter;

include("simulate_communicatingCells.jl")
include("data_processing.jl")
include("dimension_reduction.jl")
gr(dpi=300)

################################################
# Initialize for (possible) dimension reduction:
################################################
dim_reduction = "none"    # "none", "AE", "VAE"
AE_architecture = "simple"   # "simple", "deep"
latent_dim = 4
encoder = Chain()   # global initialization
################################################
# Creation and saving of plots: (requires directory "output/auto_output" in the working directory)
################################################
create_plots = true   # true, false
################################################


######################
# Main script:
######################

# Generate the simulated data (set a seed for reproducibility):
communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];
dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=7, communication_graph = communication_graph)

# Preprocess the data:
X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=0.3f0, save_figures=create_plots)

# Iterative refinement of observational units using componentwise boosting (and possibly dimension reduction):
if dim_reduction in ["AE", "VAE"]
    Z_X, Z_Y = dimension_reduction(X, Y, dim_reduction, AE_architecture, latent_dim, save_figures=create_plots)
    B, Ŷ, communication_idxs, Y_t = iterative_rematching(10, X, copy(Z_Y'), Z=copy(Z_X'), save_figures=create_plots)
else
    B, Ŷ, communication_idxs, Y_t = iterative_rematching(10, X, Y, save_figures=create_plots)
end
