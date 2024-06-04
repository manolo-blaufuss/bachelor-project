using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

using Distributions, Plots, Random, StatsBase, DataFrames, LinearAlgebra

include("simulate_communicatingCells.jl")
include("componentwise_boosting.jl")
include("extract_regressionData.jl")
include("iterative_rematching.jl")
include("mse.jl")

# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];

# Define the number of cells, genes, and groups:
# Note: This is just an example.
n_cells = 1000
n_genes = 60
n_noise_genes = 10
n_groups = 4
n_cells_per_group = n_cells ÷ n_groups

# Generate the simulated data (set a seed for reproducibility):
dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(n_cells, n_genes, n_groups; seed=7, communication_graph = communication_graph)

receptor_idxs = []
for group in keys(receptor_genes)
    for receptor in receptor_genes[group]
        push!(receptor_idxs, receptor)
    end
end
receptor_idxs = sort(receptor_idxs)

gene_idxs = [1:n_genes+n_noise_genes;]

regression_data = extract_regression_data(dataset, gene_idxs, n_cells, n_groups)

# Perform componentwise boosting using X and y from regression_data:
B = get_beta_matrix(regression_data)

# Perform iterative rematching:
X = regression_data[1]
n = 20
B_iter, Y_iter, communication_idxs, matching_coefficients = iterative_rematching(n, X, B, dataset, cell_group_assignments, n_cells, n_groups, n_cells_per_group, gene_idxs)

#mse(X, regression_data[2], B, Y_iter, B_iter, "MSE orig. B", "MSE " * string(n) * " iterations")

comm_mat = zeros(1000,1000)
for i in 1:1000
    comm_mat[communication_idxs[i], i] = 1
end
heatmap(comm_mat)