using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

using Distributions, Plots, Random, StatsBase, DataFrames, LinearAlgebra

include("simulate_communicatingCells.jl")
include("componentwise_boosting.jl")
include("data_processing.jl")
#include("iterative_rematching.jl")
#include("mse.jl")
gr(dpi=300)

# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];

# Define the number of cells, genes, and groups:
# Note: This is just an example.


# Generate the simulated data (set a seed for reproducibility):
dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=7, communication_graph = communication_graph)

X, Y = preprocess_data(dataset, group_communication_matrix, noise_percentage=0.3f0, save_figures=true)

# Perform componentwise boosting using X and y from regression_data:
#B = get_beta_matrix(regression_data)

# Perform iterative rematching:
#X = regression_data[1]
#n = 20
#B_iter, Y_iter, communication_idxs, matching_coefficients = iterative_rematching(n, X, B, dataset, cell_group_assignments, n_cells, n_groups, n_cells_per_group, gene_idxs)

#mse(X, regression_data[2], B, Y_iter, B_iter, "MSE orig. B", "MSE " * string(n) * " iterations")

B, Ŷ, communication_idxs, Y_t = iterative_rematching(10, X, Y, save_figures=true)