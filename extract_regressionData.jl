function highest_interaction_matching(sel_group::Int, sel_communication::String, communications::Dict{Any, Any}, n_cells::Int, n_groups::Int)
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
    return highest_interaction_pairs
end

function standardize(X::AbstractArray; corrected_std::Bool=true, dims::Int=1)
    return (X .- mean(X, dims=dims))./ std(X, corrected=corrected_std, dims=dims)   
end

function get_X_y(dataset::Matrix{Float32}, sel_receptors::Vector{Any}, communications::Dict{Any, Any}, sel_group::Int, sel_communication::String, n_cells::Int, n_groups::Int)
    # Define the data matrix X and the response vector y:
    sel_receptor = sel_receptors[sel_group]
    receptor_expression = dataset[:, sel_receptor]
    n_cells_per_group = n_cells ÷ n_groups
    start_cell = (sel_group - 1)*n_cells_per_group + 1
    end_cell = start_cell + n_cells_per_group - 1
    y = receptor_expression[start_cell:end_cell]

    highest_interaction_pairs = highest_interaction_matching(sel_group, sel_communication, communications, n_cells, n_groups)
    X = zeros(n_cells_per_group, size(dataset)[2])
    for i in 1:n_cells_per_group
        for j in 1: size(dataset)[2]
            X[i, j] = dataset[highest_interaction_pairs[i], j]
        end
    end
    return X, y
end

function extract_regression_data(dataset::Matrix{Float32}, sel_receptors::Vector{Any}, communications::Dict{Any, Any}, n_cells::Int, n_groups::Int)
    communications_keys = keys(communications)
    regression_data = Dict()
    for sel_communication in communications_keys
        sel_group = parse(Int, sel_communication[13])
        X, y = get_X_y(dataset, sel_receptors, communications, sel_group, sel_communication, n_cells, n_groups)
        regression_data[sel_communication] = (standardize(X), y)
    end
    return regression_data
end