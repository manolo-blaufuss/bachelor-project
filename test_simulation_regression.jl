using StatsBase

include("simulate_communicatingCells.jl")
include("componentwise_boosting.jl")
include("extract_regressionData.jl")

# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];

# Define the number of cells, genes, and groups:
# Note: This is just an example.
n_cells = 1000
n_genes = 60
n_groups = 4
n_cells_per_group = n_cells ÷ n_groups

# Generate the simulated data (set a seed for reproducibility):
dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands, communications = simulate_interacting_singleCells(n_cells, n_genes, n_groups; seed=7, communication_graph = communication_graph)

regression_data = extract_regression_data(dataset, receptor_genes, n_cells, n_groups)

# Perform componentwise boosting using X and y from regression_data:
B = get_beta_matrix(regression_data)
println(findall(x->x!=0, B))

# Calculate X * B:
X = regression_data[1]
R = X * B


receptor_idxs = []
for group in keys(receptor_genes)
    for receptor in receptor_genes[group]
        push!(receptor_idxs, receptor)
    end
end
receptor_idxs = sort(receptor_idxs)


communication_idxs = zeros(Int, n_cells)
for i in 1:n_cells
    expression_R = R[i, :]

    sender_group = parse(Int, cell_group_assignments[i][7])
    sender_start_idx = (sender_group - 1) * n_cells_per_group + 1
    sender_end_idx = sender_start_idx + n_cells_per_group - 1
    sample_idxs = setdiff(1:n_cells, sender_start_idx:sender_end_idx)
    distances = zeros(Int, n_cells) .+ Inf
    for j in sample_idxs
        expression_X = X[j, receptor_idxs]
        cosine_sim = dot(expression_X, expression_R) / (norm(expression_X) * norm(expression_R))
        distances[j] = 1 - cosine_sim
    end
    min_val = findmin(distances)[1]
    min_val_indices = findall(x -> x == min_val, distances)
    best_match = rand(min_val_indices)
    communication_idxs[i] = best_match
    
end