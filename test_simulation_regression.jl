include("simulate_communicatingCells.jl")
include("componentwise_boosting.jl")
include("extract_regressionData.jl")

# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
communication_graph = [0 1 0 0; 0 0 1 0; 1 0 0 0; 1 0 0 0];

# Define the number of cells, genes, and groups:
# Note: This is just an example.
n_cells = 1000
n_genes = 60
n_groups = 4

# Generate the simulated data (set a seed for reproducibility):
dataset, group_communication_matrix, cell_group_assignments, sel_receptors, sel_ligands, communications = simulate_interacting_singleCells(n_cells, n_genes, n_groups; seed=7, communication_graph = communication_graph)

regression_data = extract_regression_data(dataset, sel_receptors, communications, n_cells, n_groups)

# Perform componentwise boosting using X and y from regression_data:
beta_vectors = get_beta_vectors(regression_data)