#--Hyperparameter:
mutable struct Hyperparameter
    zdim::Int
    epochs::Int
    batchsize::Int
    η::Float32
    λ::Float32

    # Constructor with default values
    function Hyperparameter(; zdim::Int=10, epochs::Int=50, batchsize::Int=2^9, η::Float32=0.01f0, λ::Float32=0.1f0)
        new(zdim, epochs, batchsize, η, λ)
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

#---training function for a vanilla AE:
function train_AE!(X::AbstractMatrix{<:AbstractFloat}, AE::Autoencoder; soft_clustering::Bool=false, MD::Union{Nothing, MetaData}=nothing, save_data::Bool=false, path::Union{Nothing, String}=nothing)

    if eltype(X) != Float32
        @warn "Matrix elements are not of type Float32. This may slow down the optimization process."
    end

    n = size(X, 2)

    @info "Training AE for $(AE.HP.epochs) epochs with a batchsize of $(AE.HP.batchsize), i.e., $(Int(round(AE.HP.epochs * n / AE.HP.batchsize))) update iterations."

    opt = ADAMW(AE.HP.η, (0.9, 0.999), AE.HP.λ)
    opt_state = Flux.setup(opt, (AE.encoder, AE.decoder))
    
    mean_trainlossPerEpoch = []
    @showprogress for epoch in 1:AE.HP.epochs

        loader = Flux.Data.DataLoader(X, batchsize=AE.HP.batchsize, shuffle=true) 

        batchLosses = Float32[]
        for batch in loader
             
            batchLoss, grads = Flux.withgradient(AE.encoder, AE.decoder) do enc, dec
               X̂ = dec(enc(batch))
               Flux.mse(X̂, batch)
            end
            push!(batchLosses, batchLoss)

            Flux.update!(opt_state, (AE.encoder, AE.decoder), (grads[1], grads[2]))
        end

        push!(mean_trainlossPerEpoch, mean(batchLosses))
    end

    AE.Z = encoder(X)

    if soft_clustering

        AE.Z_cluster = softmax(split_vectors(AE.Z))
        MD.obs_df[!, :Cluster] = [argmax(AE.Z_cluster[:, i]) for i in 1:size(AE.Z_cluster, 2)]

        if !isnothing(MD.Top_features)
            MD.Top_features = topFeatures_per_Cluster(AE; save_data=save_data, path=path)
        end
        
    end

    return mean_trainlossPerEpoch
end