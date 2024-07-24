# This script contains code provided by Niklas Brunn

using Random
using Statistics
using DataFrames

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
function match_cells(group_interactions, X::AbstractMatrix{<:Float32}, celltype::AbstractVector{<:String}; replace::Bool=false, sample_seed::Int=42)

    Random.seed!()

    n, p = size(X)
    X_source = zeros(Float32, n, p)
    X_target = zeros(Float32, n, p)
    matched_inds = DataFrame(Source = zeros(Int, n), Target = zeros(Int, n))
    source_inds_vec = []
    target_inds_vec = []

    for (source_type, receiver_type) in group_interactions
        source_inds = findall(x->x==source_type, celltype)
        receiver_inds = findall(x->x==receiver_type, celltype)

        n_source = length(source_inds)
        target_inds = sample(receiver_inds, n_source, replace=replace)

        X_source[source_inds, :] = X[source_inds, :]
        X_target[source_inds, :] = X[target_inds, :]

        source_inds_vec = Int.(vcat(source_inds_vec, source_inds))
        target_inds_vec = Int.(vcat(target_inds_vec, target_inds))
    end

    matched_inds[!, :Source] = source_inds_vec
    matched_inds[!, :Target] = target_inds_vec
    sort!(matched_inds, [:Source])

    X_source_st = standardize(X_source)
    X_target_st = standardize(X_target)

    return X_source, X_target, X_source_st, X_target_st, matched_inds
end

#X_norm are the scRNA-seq data with log1p-normalized counts with cells from the 4 cell groups
X_source, X_target, X_source_st, X_target_st, matched_inds = match_cells(group_interactions, X_norm, Metadata_df.Celltype; replace=true);
matched_inds