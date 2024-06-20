using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

using Distributions, Plots, Random, StatsBase, DataFrames, LinearAlgebra, Flux, ProgressMeter;

include("simulate_communicatingCells.jl")
include("extract_regressionData.jl")
include("AE.jl")
include("componentwise_boosting.jl")
include("iterative_rematching.jl")
gr(dpi=300)

# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
communication_graph = [0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0];

# Define the number of cells, genes, and groups:
# Note: This is just an example.
n_cells = 1000
n_genes = 60
n_noise_genes = 10
n_groups = 4
n_cells_per_group = n_cells ÷ n_groups
latent_dim = 4

# Generate the simulated data (set a seed for reproducibility):
dataset, group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands = simulate_interacting_singleCells(n_cells, n_genes, n_groups; seed=7, communication_graph = communication_graph)

savefig(heatmap(dataset, title="Original Data", xlabel="Genes", ylabel="Cells"), "output/auto_output/heatmap_data.png")
savefig(heatmap(dataset, title="Original Data", xlabel="Genes", ylabel="Cells"), "output/auto_output/heatmap_data.svg")

gene_idxs = [1:n_genes+n_noise_genes;]
reduced_gene_idxs = [1:latent_dim;]

regression_data = extract_regression_data(dataset, gene_idxs, n_cells, n_groups)
X = regression_data[1]
Y = regression_data[2]

savefig(heatmap(X, title="Standardized Data", xlabel="Genes", ylabel="Cells"), "output/auto_output/heatmap_X.png")
savefig(heatmap(X, title="Standardized Data", xlabel="Genes", ylabel="Cells"), "output/auto_output/heatmap_X.svg")


# Define the hyperparameters for the AE:
HP = Hyperparameter(zdim=latent_dim, epochs=20, batchsize=2^7, η=0.01f0, λ=0.0f0)
akt = tanh_fast; #relu, sigmoid, tanh_fast, tanh, ...

# Define the encoder and decoder:
# Option 1: One layer, linear:
encoder = Chain(Dense(size(X, 2), HP.zdim))
decoder = Chain(Dense(HP.zdim, size(X, 2)))
# Option 2: Three layers, tanh_fast:
#encoder = Chain(Dense(size(X, 2), 32, akt), Dense(32, HP.zdim, akt), Dense(HP.zdim, HP.zdim, akt))
#decoder = Chain(Dense(HP.zdim, 32, akt), Dense(32, size(X, 2), akt), Dense(size(X, 2), size(X, 2)))
# Option xy

# Define the AE:
AE = Autoencoder(;encoder, decoder, HP)

# Train the AE:
mean_trainlossPerEpoch = train_AE!(X', AE)

# Get the latent representation:
Z_X = AE.encoder(X')
Z_Y = AE.encoder(Y')

savefig(heatmap(Z_X, title="Latent Representation of X", xlabel="Latent Dimensions", ylabel="Cells"), "output/auto_output/Z_X.png")
savefig(heatmap(Z_X, title="Latent Representation of X", xlabel="Latent Dimensions", ylabel="Cells"), "output/auto_output/Z_X.svg")
savefig(heatmap(Z_Y, title="Latent Representation of Y", xlabel="Latent Dimensions", ylabel="Cells"), "output/auto_output/Z_Y.png")
savefig(heatmap(Z_Y, title="Latent Representation of Y", xlabel="Latent Dimensions", ylabel="Cells"), "output/auto_output/Z_Y.svg")
savefig(heatmap(abs.(cor(Z_X, dims=2)), xlabel = "Latent Dimensions", ylabel = "Latent Dimensions", title = "Absolute Correlation", color = :vik), "output/auto_output/correlation.png")
savefig(heatmap(abs.(cor(Z_X, dims=2)), xlabel = "Latent Dimensions", ylabel = "Latent Dimensions", title = "Absolute Correlation", color = :vik), "output/auto_output/correlation.svg")
savefig(plot(1:length(mean_trainlossPerEpoch), mean_trainlossPerEpoch, title = "Mean train loss per epoch", xlabel = "Epoch", ylabel = "Loss", legend = true, label = "Train loss", linecolor = :red, linewidth = 2), "output/auto_output/train_loss.png")
savefig(plot(1:length(mean_trainlossPerEpoch), mean_trainlossPerEpoch, title = "Mean train loss per epoch", xlabel = "Epoch", ylabel = "Loss", legend = true, label = "Train loss", linecolor = :red, linewidth = 2), "output/auto_output/train_loss.svg")


# Perform componentwise boosting using X and latent representation of Y:
B = get_beta_matrix((X, Z_Y'))

savefig(heatmap(B, title="Beta Matrix", xlabel="Latent Dimensions", ylabel="Genes"), "output/auto_output/B.png")
savefig(heatmap(B, title="Beta Matrix", xlabel="Latent Dimensions", ylabel="Genes"), "output/auto_output/B.svg")
savefig(scatter(mean((X * B - Z_Y').^2, dims=1)', title = "Mean Squared Error (initial) ", label = "MSE", xaxis = "Latent Dimensions", yaxis = "MSE", markersize=1), "output/auto_output/MSE_initial.png")
savefig(scatter(mean((X * B - Z_Y').^2, dims=1)', title = "Mean Squared Error (initial) ", label = "MSE", xaxis = "Latent Dimensions", yaxis = "MSE", markersize=1), "output/auto_output/MSE_initial.svg")


# Perform iterative rematching:
n = 20
B_iter, Y_iter, communication_idxs, matching_coefficients = iterative_rematching(n, X, copy(Z_X'), B, dataset, cell_group_assignments, n_cells, n_groups, n_cells_per_group, reduced_gene_idxs)
