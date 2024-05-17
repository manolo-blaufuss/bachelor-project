function standardize(X::AbstractArray; corrected_std::Bool=true, dims::Int=1)
    X = (X .- mean(X, dims=dims))./ std(X, corrected=corrected_std, dims=dims)

    # Replace NaN values with zeros:
    for i in 1:size(X)[2]
        if sum(isnan.(X[:, i])) > 0
            X[:, i] = zeros(size(X)[1])
        end
    end
    return X
end

function get_y(dataset::Matrix{Float32}, receptor_idx::Int, communication_idxs::Vector{Int})
    # Define the response vector y:
    receptor_expression = dataset[:, receptor_idx]
    y = zeros(length(receptor_expression))
    for i in 1:length(receptor_expression)
        y[i] = receptor_expression[communication_idxs[i]]
    end
    return (y .- mean(y))./ std(y, corrected=true)
end

function assign_communication_partners(n_cells::Int, n_groups::Int, communication_pairs::Vector{Any})
    communication_idxs = zeros(Int, n_cells)
    n_cells_per_group = n_cells ÷ n_groups
    threshold = round(Int, 0.7 * n_cells_per_group)
    for sel_communication in communication_pairs
        sender_group = sel_communication[1]
        receiver_group = sel_communication[2]
        sender_start_idx = (sender_group - 1) * n_cells_per_group + 1
        sender_end_idx = sender_start_idx + n_cells_per_group - 1
        receiver_start_idx = (receiver_group - 1) * n_cells_per_group + 1
        # receiver_end_idx = receiver_start_idx + n_cells_per_group - 1
        sample_idxs = setdiff(1:n_cells, sender_start_idx:sender_end_idx)
        for i in 0:n_cells_per_group-1
            if i < threshold
                communication_idxs[sender_start_idx + i] = receiver_start_idx + i
            else
                communication_idxs[sender_start_idx + i] = rand(sample_idxs)
            end
        end
    end
    return communication_idxs
end

function extract_regression_data(dataset::Matrix{Float32}, receptor_genes::Dict{Any, Any}, n_cells::Int, n_groups::Int)
    # Get communication pairs:
    communication_pairs = []
    for i in 1:n_groups
        for j in 1:n_groups
            if communication_graph[i, j] == 1
                push!(communication_pairs, (i, j))
            end
        end
    end
    
    # Assign communication partners for each cell:
    communication_idxs = assign_communication_partners(n_cells, n_groups, communication_pairs)
    
    # Set X:
    X = standardize(dataset)

    # Get Matrix containing y for each receptor gene:
    receptor_idxs = []
    for group in keys(receptor_genes)
        for receptor in receptor_genes[group]
            push!(receptor_idxs, receptor)
        end
    end
    receptor_idxs = sort(receptor_idxs)
    Y = zeros(n_cells, length(receptor_idxs))
    for i in 1:length(receptor_idxs)
        Y[:, i] = get_y(dataset, receptor_idxs[i], communication_idxs)
    end


    return X, Y
end