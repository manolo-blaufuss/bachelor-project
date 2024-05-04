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

# Perform componentwise boosting using X and y:

ϵ = 0.1 #learning rate (step width) for the boosting 
M = 2 #number of boosting steps 
#β = zeros(size(X, 2)) #start with a zero initialization of the coefficient vector

#compL2Boost!(β, X, y, ϵ, M) 

# Display the learned coefficients:
#println(β)
#println("\nMean Squared Error compBoost:")
#println(mean((y-X*β).^2))