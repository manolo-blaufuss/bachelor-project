include("simulate_communicatingCells.jl")
include("componentwise_boosting.jl")

# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
communication_graph = [1 0 0 0; 0 1 0 0; 0 1 0 0; 0 0 1 0];

# Define the number of cells, genes, and groups:
# Note: This is just an example.
n_cells = 1000
n_genes = 60
n_groups = 4

# Generate the simulated data (set a seed for reproducibility):
dataset, group_communication_matrix, cell_group_assignments, sel_receptors, sel_ligands, communications = simulate_interacting_singleCells(n_cells, n_genes, n_groups; seed=7, communication_graph = communication_graph)


# Matching of cells with highest interaction for a given group
sel_group = 1
sel_communication = "Group1-Group1"
interaction_df = communications[sel_communication]

n_cells_per_group = n_cells ÷ n_groups

highest_interaction_pairs = zeros(n_cells_per_group)
for i in 1:n_cells_per_group
    start_row = ((sel_group - 1)*n_cells*n_cells_per_group)+(i-1)*n_cells+1
    end_row = start_row + n_cells - 1
    highest_interaction_row = argmax(interaction_df[start_row : end_row, :GeometricMean])
    highest_interaction_pairs[i] = interaction_df[highest_interaction_row, :Sender]
end
highest_interaction_pairs = Int.(highest_interaction_pairs)

# Define the data matrix X and the response vector y:
sel_receptor = sel_receptors[sel_group]
sel_ligand = sel_ligands[sel_group]
receptor_expression = dataset[:, sel_receptor]
start_cell = (sel_group - 1)*n_cells_per_group + 1
end_cell = start_cell + n_cells_per_group - 1
y = receptor_expression[start_cell:end_cell]

X = zeros(n_cells_per_group, size(dataset)[2])
for i in 1:n_cells_per_group
    for j in 1: size(dataset)[2]
        X[i, j] = dataset[highest_interaction_pairs[i], j]
    end
end