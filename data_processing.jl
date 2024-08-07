########################
# Help Functions
########################

include("componentwise_boosting.jl")

function get_y(X::Matrix{Float32}, idx::Int, communication_idxs::Vector{Int})
    # Define the response vector y:
    idx_expression = X[:, idx]
    y = zeros(length(idx_expression))
    for i in 1:length(idx_expression)
        y[i] = idx_expression[communication_idxs[i]]
    end
    return y
end

#######################################
# Data Processing: Standalone Functions
#######################################

function get_beta_matrix(X, Y; ϵ = 0.2, M = 7, save_figures::Bool=false, iter::Int=-1)
    beta_matrix = zeros(Float32, size(X, 2), size(Y, 2))
    for i in 1:size(Y, 2)
        y = Y[:, i]
        β = zeros(size(X, 2)) #start with a zero initialization of the coefficient vector
        compL2Boost!(β, X, y, ϵ, M)
        beta_matrix[:, i] = β
    end
    if save_figures
        filename = "beta_matrix_" * string(iter)
        # Create a heatmap of the beta matrix and save it:
        savefig(heatmap(beta_matrix, title="Beta Matrix (iteration " * string(iter) * ")", xlabel="Predictors", ylabel="Responses"), "output/auto_output/" * filename * ".png")
        savefig(heatmap(beta_matrix, title="Beta Matrix (iteration " * string(iter) * ")", xlabel="Predictors", ylabel="Responses"), "output/auto_output/" * filename * ".svg")
    end
    return beta_matrix
end

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

function init_communication_partners(n_cells::Int, n_groups::Int, communication_pairs::Vector{Tuple{Int,Int}}, noise_percentage::Float32)
    communication_idxs = zeros(Int, n_cells)
    n_cells_per_group = n_cells ÷ n_groups
    threshold = round(Int, (1-noise_percentage) * n_cells_per_group)
    for sel_communication in communication_pairs
        sender_group = sel_communication[1]
        receiver_group = sel_communication[2]
        sender_start_idx = (sender_group - 1) * n_cells_per_group + 1
        receiver_start_idx = (receiver_group - 1) * n_cells_per_group + 1
        receiver_end_idx = receiver_start_idx + n_cells_per_group - 1
        sample_idxs = setdiff(1:n_cells, receiver_start_idx:receiver_end_idx)
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

function get_Y(X::Matrix{Float32}, communication_idxs::Vector{Int}; save_figures::Bool=false, iter::Int=-1)
    Y = zeros(Float32, size(X))
    for i in 1:size(X)[2]
        Y[:, i] = get_y(X, i, communication_idxs)
    end
    if save_figures
        filename = "Y_" * string(iter)
        h = heatmap(Y, title="Response Matrix (iteration " * string(iter) * ")", xlabel="Features", ylabel="Observation Units")
        savefig(h, "output/auto_output/" * filename * ".png")
        savefig(h, "output/auto_output/" * filename * ".svg")
    end
    return Y
end

function get_prediction(X::Matrix{Float32}, B::Matrix{Float32}; save_figures::Bool=false, iter::Int=-1)
    Ŷ = X * B
    if save_figures
        filename = "prediction_" * string(iter)
        h = heatmap(Ŷ, title="Predicted Response Matrix (iteration " * string(iter) * ")", xlabel="Features", ylabel="Observation Units")
        savefig(h, "output/auto_output/" * filename * ".png")
        savefig(h, "output/auto_output/" * filename * ".svg")
    end
    return Ŷ
end

function refine_interaction_partners(X::Matrix{Float32}, Ŷ::Matrix{Float32}; save_figures::Bool=false, iter::Int=-1)
    n_cells, n_genes = size(X)
    communication_idxs = zeros(Int, n_cells)
    
    # Sample two random cells and following cell to plot the cosine similarity:
    sample_cells = rand(1:n_cells, 2)
    push!(sample_cells, sample_cells[1] + 1)
    push!(sample_cells, sample_cells[2] + 1)
    
    for i in 1:n_cells
        expression_Ŷ = Ŷ[i, :]
        cosine_similarities = zeros(n_cells)
        for j in 1:n_cells
            expression_X = X[j, 1:n_genes]
            cosine_sim = dot(expression_X, expression_Ŷ) / (norm(expression_X) * norm(expression_Ŷ))
            cosine_similarities[j] = cosine_sim
        end
        sorted_sims = sort(cosine_similarities)
        sim_distances = diff(sorted_sims)
        max_distance = findmax(sim_distances)[1]
        max_similarity = sorted_sims[n_cells]
        similarity_threshold = max_similarity - 0.3 * max_distance
        sel_indices = findall(x -> x >= similarity_threshold, cosine_similarities)
        communication_idxs[i] = rand(sel_indices)
        if save_figures && i in sample_cells
            #s = scatter(cosine_similarities, title="Cosine Similarities (unit " *string(i)* ", iteration " * string(iter) * ")", xlabel="Cells", ylabel="Cosine Similarity", markersize=1, markerstrokewidth=0.1, legend=false)
            #savefig(s, "output/auto_output/cosine_similarities_" *string(i) *"_"* string(iter) * ".png")
            #savefig(s, "output/auto_output/cosine_similarities_" *string(i) *"_"* string(iter) * ".svg")
        end
    end
    if save_figures
        s = scatter(1:n_cells, communication_idxs, title="Matching partners (iteration " * string(iter) * ")", xlabel="Sender units", ylabel="Receiver units", markersize=1, markerstrokewidth=0.1, legend=false)
        savefig(s, "output/auto_output/matching_partners_" * string(iter) * ".png")
        savefig(s, "output/auto_output/matching_partners_" * string(iter) * ".svg")
    end
    return communication_idxs
end

#######################################
# Data Processing: Main Functions
#######################################

"""
    preprocess_data(dataset::Matrix{Float32}, communication_graph::Matrix{Int}; noise_percentage::Float32=0.3f0, save_figures::Bool=false)

Preprocess the data by standardizing the dataset, assigning initial communication partners for each cell, and getting the response matrix Y.

# Arguments
- `dataset::Matrix{Float32}`: The dataset of size n x p.
- `communication_graph::Matrix{Int}`: The communication graph of size n_groups x n_groups with 1 indicating communication between groups.
- `noise_percentage::Float32`: The percentage of noise in the initial communication partners.
- `save_figures::Bool`: Whether to save the figures.

# Returns
- `X::Matrix{Float32}`: The standardized dataset.
- `Y::Matrix{Float32}`: The response matrix Y.
"""
function preprocess_data(dataset::Matrix{Float32}, communication_graph::Matrix{Int}; noise_percentage::Float32=0.3f0, save_figures::Bool=false)
    n_cells = size(dataset)[1]
    n_groups = size(communication_graph)[1]
    
    # Get communication pairs:
    communication_pairs = Tuple{Int,Int}[]
    for i in 1:n_groups
        for j in 1:n_groups
            if communication_graph[i, j] == 1
                push!(communication_pairs, (i, j))
            end
        end
    end

    # Set X:
    X = standardize(dataset)
    
    # Assign initial communication partners for each cell:
    communication_idxs = init_communication_partners(n_cells, n_groups, communication_pairs, noise_percentage)

    # Get Matrix containing y for each receptor gene:
    Y = get_Y(X, communication_idxs)

    # Plot and save figures of the original data, X, Y, communication_idxs if save_figures==true:
    if save_figures
        h1 = heatmap(dataset, title="Simulated Data", xlabel="Features", ylabel="Observation Units")
        h2 = heatmap(X, title="Standardized Data", xlabel="Features", ylabel="Observation Units")
        h3 = heatmap(Y, title="Response Matrix t=0", xlabel="Features", ylabel="Observation Units")
        s = scatter(1:n_cells, communication_idxs, title="Initial matching partners", xlabel="Sender units", ylabel="Receiver units", markersize=1, markerstrokewidth=0.1, legend=false)
        savefig(h1, "output/auto_output/dataset.png")
        savefig(h1, "output/auto_output/dataset.svg")
        savefig(h2, "output/auto_output/X.png")
        savefig(h2, "output/auto_output/X.svg")
        savefig(h3, "output/auto_output/Y_0.png")
        savefig(h3, "output/auto_output/Y_0.svg")
        savefig(s, "output/auto_output/matching_partners_0.png")
        savefig(s, "output/auto_output/matching_partners_0.svg")
    end
    return X, Y
end


"""
    iterative_rematching(n::Int, X::Matrix{Float32}, Y::Matrix{Float32}; Z::Union{Matrix{Float32}, Nothing} = nothing, save_figures::Bool=false, groups::Union{AbstractArray, Nothing} = nothing)

Iteratively refine the mapping of observational units using componentwise boosting.

# Arguments
- `n::Int`: The number of iterations.
- `X::Matrix{Float32}`: The standardized dataset of size n x p.
- `Y::Matrix{Float32}`: The response matrix Y.
- `Z::Union{Matrix{Float32}, Nothing}`: The latent representation (if applicable) of the dataset.
- `save_figures::Bool`: Whether to save the figures.
- `groups::Union{AbstractArray, Nothing}`: The groups for convergence checking.

# Returns
- `B::Matrix{Float32}`: The beta matrix.
- `Ŷ::Matrix{Float32}`: The predicted response matrix.
- `communication_idxs::Vector{Int}`: The communication partners.
- `Y_t::Matrix{Float32}`: The response matrix Y after the last iteration.
- `matching_coefficient::Float32`: The matching coefficient (percentage of correct assignments). Important: Check the correct assignment of groups in the function.
"""
function iterative_rematching(n::Int, X::Matrix{Float32}, Y::Matrix{Float32}; Z::Union{Matrix{Float32}, Nothing} = nothing, save_figures::Bool=false, groups::Union{AbstractArray, Nothing} = nothing)
    if isnothing(Z)
        Z = X
    else
        Z = Z
    end
    Y_t = Y
    B = zeros(Float32, size(X)[2], size(Y_t)[2])
    Ŷ = zeros(Float32, size(Z)[1], size(B)[2])
    communication_idxs = zeros(Int, size(Z)[1])
    communication_idxs_last = zeros(Int, size(Z)[1])
    matching_coefficient = 0

    mse = []

    for t in 1:n
        if save_figures && (t == 1 || t % 5 == 0 || t == n)
            save_figures_t = true
        else
            save_figures_t = false
        end
        save_figures_t = save_figures
        # Perform cmponentwise boosting
        B = get_beta_matrix(X, Y_t, save_figures=save_figures_t, iter=t)

        # Get prediction:
        Ŷ = get_prediction(X, B, save_figures=save_figures_t, iter=t)
        
        # Get communication partners using cosine similarity:
        communication_idxs = refine_interaction_partners(Z, Ŷ, save_figures=save_figures_t, iter=t)
        Y_t = get_Y(Z, communication_idxs, save_figures=save_figures_t, iter=t)

        if save_figures_t
            push!(mse, (t, mean((Y_t - Ŷ).^2, dims=1)'))
        end

        # Check convergence if groups given:
        #println(count(x -> 251 <= x <= 500, communication_idxs[1:250])+count(x -> 501 <= x <= 750, communication_idxs[251:500])+count(x -> 751 <= x <= 1000, communication_idxs[501:750])+count(x -> 1 <= x <= 250, communication_idxs[751:1000]))
        #println(count(x -> 1794 <= x <= 2087, communication_idxs[1:1228])+count(x -> 1229 <= x <= 1309, communication_idxs[1229:1309])+count(x -> 1794 <= x <= 2087, communication_idxs[1310:1793])+count(x -> 1794 <= x <= 2087, communication_idxs[1794:2087]))
        #println(count(x -> 1794 <= x <= 2087, communication_idxs)+count(x -> 1229 <= x <= 1309, communication_idxs))        
        if !isnothing(groups)
            convergence = true
            for i in 1:length(communication_idxs)
                for group in groups
                    if communication_idxs[i] in group && !(communication_idxs_last[i] in group)
                        convergence = false
                    end
                end
            end
            if convergence
                println("Convergence reached after iteration ", t)
                #break
            end
            communication_idxs_last = copy(communication_idxs)
        end
        if t == n
            matching_count = count(x -> 251 <= x <= 500, communication_idxs[1:250])+count(x -> 501 <= x <= 750, communication_idxs[251:500])+count(x -> 751 <= x <= 1000, communication_idxs[501:750])+count(x -> 1 <= x <= 250, communication_idxs[751:1000])
            #matching_count = count(x -> 1794 <= x <= 2087, communication_idxs[1:1228])+count(x -> 1229 <= x <= 1309, communication_idxs[1229:1309])+count(x -> 1794 <= x <= 2087, communication_idxs[1310:1793])+count(x -> 1794 <= x <= 2087, communication_idxs[1794:2087])
            matching_coefficient = matching_count / length(communication_idxs)
        end
    end
    # Plot MSEs if save_figures==true:
    if save_figures
        plot(mse[1][2], label="t="*string(mse[1][1]), title="Mean Squared Error", xlabel="Features", ylabel="MSE")
        for i in 2:length(mse)
            plot!(mse[i][2], label="t="*string(mse[i][1]))
        end
        savefig("output/auto_output/mse.png")
        savefig("output/auto_output/mse.svg")
    end
    return B, Ŷ, communication_idxs, Y_t, matching_coefficient
end