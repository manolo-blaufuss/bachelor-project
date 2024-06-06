using Flux;
using Random; 
using Statistics;
using DelimitedFiles;
using Plots;
using LinearAlgebra;
using DataFrames;
using VegaLite;
using UMAP;
using ProgressMeter;
using ColorSchemes;
using CSV;

#--Hyperparameter:
mutable struct Hyperparameter
    zdim::Int
    epochs::Int
    batchsize::Int
    η::Float32
    λ::Float32
    ϵ::Float32
    M::Int

    # Constructor with default values
    function Hyperparameter(; zdim::Int=10, epochs::Int=50, batchsize::Int=2^9, η::Float32=0.01f0, λ::Float32=0.1f0, ϵ::Float32=0.001f0, M::Int=1)
        new(zdim, epochs, batchsize, η, λ, ϵ, M)
    end 
end

mutable struct MetaData
    obs_df::DataFrame
    featurename::Union{Nothing, Vector{String}}
    Top_features::Union{Nothing, Dict{String, DataFrame}}

    # Constructor with default values
    function MetaData(; obs_df::DataFrame=DataFrame(), featurename::Union{Nothing, Vector{String}}=nothing, Top_features::Union{Nothing, Dict{String, DataFrame}}=nothing)
        new(obs_df, featurename, Top_features)
    end 
end

#---AE architecture:
mutable struct Autoencoder
    encoder::Union{Chain, Dense}
    decoder::Union{Chain, Dense}
    HP::Hyperparameter
    Z::Union{Nothing, Matrix{Float32}}
    Z_cluster::Union{Nothing, Matrix{Float32}}
    UMAP::Union{Nothing, Matrix{Float32}}

    # Constructor to allow initializing Z as nothing
    function Autoencoder(; encoder::Union{Chain, Dense}, decoder::Union{Chain, Dense}, HP::Hyperparameter)
        new(encoder, decoder, HP, nothing, nothing, nothing)
    end
end
(AE::Autoencoder)(X) = AE.decoder(AE.encoder(X))
Flux.@functor Autoencoder

function Base.summary(AE::Autoencoder)
    HP = AE.HP
    println("Initial hyperparameter for constructing and training an AE:
     latent dimensions: $(HP.zdim),
     training epochs: $(HP.epochs),
     batchsize: $(HP.batchsize),
     learning rate for decoder parameter: $(HP.η),
     weight decay parameter for decoder parameters: $(HP.λ)."
    )
end