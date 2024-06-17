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
X = regression_data[1]
Y = regression_data[2]


# Define the hyperparameters for the AE:
HP = Hyperparameter(zdim=latent_dim, epochs=20, batchsize=2^7, η=0.01f0, λ=0.0f0)
akt = tanh_fast; #relu, sigmoid, tanh_fast, tanh, ...

# Define the encoder and decoder:
modelseed = 42;
Random.seed!(modelseed)
# Option 1: One layer, linear:
#encoder = Chain(Dense(size(X, 2), 2 * HP.zdim))
#decoder = Chain(Dense(HP.zdim, 2 * size(X, 2)))
# Option 2: Three layers, tanh_fast:
#encoder = Chain(Dense(size(X, 2), 32, akt), Dense(32, 2*HP.zdim, akt), Dense(2*HP.zdim, 2*HP.zdim, akt))
#decoder = Chain(Dense(HP.zdim, 32, akt), Dense(32, 2*size(X, 2), akt), Dense(2*size(X, 2), 2*size(X, 2)))
# Option 3
encoder = Chain(Dense(size(X, 2) => 2^9, akt), Dense(2^9 => 2^7, akt), Dense(2^7 => 2*HP.zdim, akt), Dense(2*HP.zdim => 2*HP.zdim));
decoder = Chain(Dense(HP.zdim => 2^7, akt), Dense(2^7 => 2^9, akt), Dense(2^9 => 2*size(X,2), akt), Dense(2*size(X,2) => 2*size(X,2))); #2*p with trainable variance and p without!


# Define the AE:
AE = Autoencoder(;encoder, decoder, HP)
summary(AE)

# Train the AE:
mean_trainlossPerEpoch = train_gaußianVAE!(X', AE)

# Get the latent representation:
#1) Get the mean of the input data: 
Z_μ_X = get_VAE_latentRepresentation(AE.encoder, X'; sampling=false)[1];
Z_μ_Y = get_VAE_latentRepresentation(AE.encoder, Y'; sampling=false)[1];
#2) Get the variance of the input data:
Z_logvar = get_VAE_latentRepresentation(AE.encoder, X'; sampling=false)[2];
#3) Get the (sampled) latent representation of the input data:
Z_X = get_VAE_latentRepresentation(AE.encoder, X'; sampling=true)[3];
Z_Y = get_VAE_latentRepresentation(AE.encoder, Y'; sampling=true)[3];


# Perform componentwise boosting using X and latent representation of Y:
B = get_beta_matrix((X, Z_Y'))

# Perform iterative rematching:
n = 10
B_iter, Y_iter, communication_idxs, matching_coefficients = iterative_rematching(n, X, copy(Z_X'), B, dataset, cell_group_assignments, n_cells, n_groups, n_cells_per_group, reduced_gene_idxs)
