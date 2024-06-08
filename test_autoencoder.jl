using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

using Distributions, Plots, Random, StatsBase, DataFrames, LinearAlgebra, Flux, ProgressMeter, Plots;

include("simulate_communicatingCells.jl")
include("extract_regressionData.jl")
include("AE.jl")
include("componentwise_boosting.jl")
include("iterative_rematching.jl")

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

gene_idxs = [1:n_genes+n_noise_genes;]
reduced_gene_idxs = [1:latent_dim;]

regression_data = extract_regression_data(dataset, gene_idxs, n_cells, n_groups)
X = regression_data[1]'


# Define the hyperparameters for the AE:
HP = Hyperparameter(zdim=latent_dim, epochs=20, batchsize=2^9, η=0.01f0, λ=0.1f0)

# Define the encoder and decoder:
encoder = Chain(Dense(size(X, 1), 64, tanh_fast), Dense(64, 32, tanh_fast), Dense(32, HP.zdim))
decoder = Chain(Dense(HP.zdim, 32, tanh_fast), Dense(32, 64, tanh_fast), Dense(64, size(X, 1), identity))

# Define the AE:
AE = Autoencoder(;encoder, decoder, HP)

# Train the AE:
mean_trainlossPerEpoch = train_AE!(X, AE)

# Get the latent representation:
Z = AE.encoder(X)
Z = copy(Z')

regression_data_reduced = extract_regression_data(Z, reduced_gene_idxs, n_cells, n_groups)

# Perform componentwise boosting using X and y from regression_data:
X_reduced = regression_data_reduced[1]
Y_reduced = regression_data_reduced[2]

B = get_beta_matrix(regression_data_reduced)

# Perform iterative rematching:
n = 20
B_iter, Y_iter, communication_idxs, matching_coefficients = iterative_rematching(n, Z, B, dataset, cell_group_assignments, n_cells, n_groups, n_cells_per_group, reduced_gene_idxs)
