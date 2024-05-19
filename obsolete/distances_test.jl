# Install necessary packages
using Pkg
Pkg.add("Distances")
Pkg.add("StatsBase")
Pkg.add("LinearAlgebra")

# Load packages
using Distances
using StatsBase
using LinearAlgebra

# Generate two random vectors of length 5
a = rand(5)
b = rand(5)

# Compute Euclidean distance using Distances.jl
euclidean_dist_distances = evaluate(Euclidean(), a, b)[1] #use: pairwise(Euclidean(), X, Y, dims=1) for computing the pairwise euclidean dist between each row of X and each row of Y.

# Compute Cosine similarity using Distances.jl
cosine_sim_distances = 1 - evaluate(CosineDist(), a, b)[1] #use: pairwise(CosineDist(), X, Y, dims=1) for computing the pairwise euclidean dist between each row of X and each row of Y.

# Compute Euclidean distance using StatsBase.jl
euclidean_dist_statsbase = euclidean(a, b) #is equal to: sqrt(sum((a.-b).^2))

# Compute Cosine similarity using StatsBase.jl
cosine_sim_statsbase = dot(a, b) / (norm(a) * norm(b))

# Print results
println("Vector a: ", a)
println("Vector b: ", b)
println()
println("Euclidean distance (Distances.jl): ", euclidean_dist_distances)
println("Cosine similarity (Distances.jl): ", cosine_sim_distances)
println()
println("Euclidean distance (StatsBase.jl): ", euclidean_dist_statsbase)
println("Cosine similarity (StatsBase.jl): ", cosine_sim_statsbase)