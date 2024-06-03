"""
    iterative_rematching(n::Int, X::Matrix{Float32}, B::Matrix{}, dataset::Matrix{Float32}, cell_group_assignments::Vector{String}, n_cells::Int, n_groups::Int, n_cells_per_group::Int, gene_idxs::Vector{Int})

Perform iterative rematching for communication between cells.

# Arguments
- `n::Int`: Number of iterations.
- `X::Matrix{Float32}`: Design matrix.
- `B::Matrix{}`: Beta matrix.
- `dataset::Matrix{Float32}`: The input dataset.
- `cell_group_assignments::Vector{String}`: The cell group assignments.
- `n_cells::Int`: The number of cells.
- `n_groups::Int`: The number of groups.
- `n_cells_per_group::Int`: The number of cells per group.
- `gene_idxs::Vector{Int}`: The indices of the genes to be considered.

# Returns
- `B::Matrix{Float32}`: The updated beta matrix.
- `Y::Matrix{Float32}`: The response matrix Y.
- `communication_idxs::Vector{Int}`: The indices of the communication partners.
- `matching_coefficients::Dict`: The matching coefficients.
"""
function iterative_rematching(n::Int, X::Matrix{Float32}, B::Matrix{}, dataset::Matrix{Float32}, cell_group_assignments::Vector{String}, n_cells::Int, n_groups::Int, n_cells_per_group::Int, gene_idxs::Vector{Int})
    Y = zeros(n_cells, length(receptor_idxs))
    communication_idxs = zeros(Int, n_cells)
    for iter in 1:n
        R = X * B
        communication_idxs = zeros(Int, n_cells)
        for i in 1:n_cells
            expression_R = R[i, :]
            #sender_group = parse(Int, cell_group_assignments[i][7])
            #sender_start_idx = (sender_group - 1) * n_cells_per_group + 1
            #sender_end_idx = sender_start_idx + n_cells_per_group - 1
            # sample_idxs = setdiff(1:n_cells, sender_start_idx:sender_end_idx)
            cosine_similarities = zeros(n_cells)
            for j in 1:n_cells
                expression_X = X[j, gene_idxs]
                cosine_sim = dot(expression_X, expression_R) / (norm(expression_X) * norm(expression_R))
                cosine_similarities[j] = cosine_sim
            end
            sorted_sims = sort(cosine_similarities)
            sim_distances = diff(sorted_sims)
            max_distance = findmax(sim_distances)[1]
            max_similarity = sorted_sims[n_cells]
            similarity_threshold = max_similarity - 0.2 * max_distance
            sel_indices = findall(x -> x >= similarity_threshold, cosine_similarities)
            communication_idxs[i] = rand(sel_indices)

            if i in [101, 102] && iter == n
                plot_similarity(cosine_similarities, sorted_sims, i, iter)
            end
        end
        # Get Matrix containing y for each receptor gene:
        Y = zeros(n_cells, length(gene_idxs))
        for i in 1:length(gene_idxs)
            Y[:, i] = get_y(dataset, gene_idxs[i], communication_idxs)
        end
        # Perform componentwise boosting using X and y from regression_data:
        B = get_beta_matrix((X, Y))
    end
    # Get coefficients that measure how many cells are matched correctly:
    communication_pairs = []
    for i in 1:n_groups
        for j in 1:n_groups
            if communication_graph[i, j] == 1
                push!(communication_pairs, (i, j))
            end
        end
    end

    matching_coefficients = Dict()
    for pair in communication_pairs
        sender_start_idx = (pair[1] - 1) * n_cells_per_group + 1
        sender_end_idx = sender_start_idx + n_cells_per_group - 1
        receiver_start_idx = (pair[2] - 1) * n_cells_per_group + 1
        receiver_end_idx = receiver_start_idx + n_cells_per_group - 1
        matching_coefficients[pair] = count(x -> receiver_start_idx <= x <= receiver_end_idx, communication_idxs[sender_start_idx:sender_end_idx]) / 250
    end
    return B, Y, communication_idxs, matching_coefficients
end

function plot_similarity(sims, sorted_sims, cell_number, iter)
    p = plot(sims, title = "Cosine sims cell " * string(cell_number) * " iter " * string(iter), label = "Similarity", xaxis = "Index", yaxis = "Similarity")
    display(p)
    q = plot(sorted_sims, title = "Sorted cosine sims cell " * string(cell_number) * " iter " * string(iter), label = "Similarity", xaxis = "Index", yaxis = "Similarity")
    display(q)
end