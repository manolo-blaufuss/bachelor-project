"""
    iterative_rematching(n::Int, X::Matrix{Float32}, B::Matrix{Any}, dataset::Matrix{Float32}, cell_group_assignments::Vector{String}, n_cells::Int, n_cells_per_group::Int, receptor_idxs::Vector{Any})

Perform iterative rematching for communication between cells.

# Arguments
- `n::Int`: Number of iterations.
- `X::Matrix{Float32}`: Design matrix.
- `B::Matrix{Any}`: Beta matrix.
- `dataset::Matrix{Float32}`: The input dataset.
- `cell_group_assignments::Vector{String}`: The cell group assignments.
- `n_cells::Int`: The number of cells.
- `n_cells_per_group::Int`: The number of cells per group.
- `receptor_idxs::Vector{Any}`: The indices of the receptor genes.

# Returns
- `B::Matrix{Float32}`: The updated beta matrix.
- `Y::Matrix{Float32}`: The response matrix Y.
- `communication_idxs::Vector{Int}`: The indices of the communication partners.
"""
function iterative_rematching(n::Int, X::Matrix{Float32}, B::Matrix{Any}, dataset::Matrix{Float32}, cell_group_assignments::Vector{String}, n_cells::Int, n_cells_per_group::Int, receptor_idxs::Vector{Any})
    Y = zeros(n_cells, length(receptor_idxs))
    communication_idxs = zeros(Int, n_cells)
    for iter in 1:n
        R = X * B
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
            communication_idxs[i] = rand(min_val_indices)
        end
        # Get Matrix containing y for each receptor gene:
        Y = zeros(n_cells, length(receptor_idxs))
        for i in 1:length(receptor_idxs)
            Y[:, i] = get_y(dataset, receptor_idxs[i], communication_idxs)
        end
        # Perform componentwise boosting using X and y from regression_data:
        B = get_beta_matrix((X, Y))
    end
    return B, Y, communication_idxs
end
