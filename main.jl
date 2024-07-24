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
# Initialize which data to use:
################################################
use_data = "simulated"    # "simulated", "real"
################################################
# Initialize for (possible) dimension reduction:
################################################
dim_reduction = "none"    # "none", "VAE", "VAE_fixed"
latent_dim = 4
################################################
# Creation and saving of plots: (requires directory "output/auto_output" in the working directory)
################################################
create_plots = false    # true, false
################################################


######################
# Main script:
######################

if use_data == "real"
    include("data/load_data.jl")
    X = standardize(X_source)
    Y = standardize(X_target)
    groups = [1:1228, 1229:1309, 1310:1793, 1794:2087]
    Random.seed!(3)
else
    # Generate the simulated data (set a seed for reproducibility):
    communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];
    dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=3, communication_graph = communication_graph)
    # Preprocess the data:
    X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=0.7f0, save_figures=create_plots)
    groups = [1:250, 251:500, 501:750, 751:1000]
end

# Iterative refinement of the mapping of observational units using componentwise boosting (and possibly dimension reduction):
if dim_reduction in ["VAE", "VAE_fixed"]
    Z_X, Z_Y = dimension_reduction(X, Y, dim_reduction, latent_dim, save_figures=create_plots, seed=5)
    B, Ŷ, communication_idxs, Y_t, matching_coefficient = iterative_rematching(10, X, copy(Z_Y'), Z=copy(Z_X'), groups=groups, save_figures=create_plots)
else
    B, Ŷ, communication_idxs, Y_t, matching_coefficient = iterative_rematching(10, X, Y, groups=groups, save_figures=create_plots)
end
