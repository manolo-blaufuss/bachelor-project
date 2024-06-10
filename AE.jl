#--Hyperparameter:
mutable struct Hyperparameter
    zdim::Int # number of latent dimensions
    epochs::Int # number of training epochs
    batchsize::Int # batchsize for training
    η::Float32 # learning rate
    λ::Float32 # weight decay parameter

    # Constructor with default values
    function Hyperparameter(; zdim::Int=10, epochs::Int=50, batchsize::Int=2^9, η::Float32=0.01f0, λ::Float32=0.1f0)
        new(zdim, epochs, batchsize, η, λ)
    end 
end

#---AE architecture:
mutable struct Autoencoder
    encoder::Union{Chain, Dense} # encoder
    decoder::Union{Chain, Dense} # decoder
    HP::Hyperparameter # hyperparameters
    Z::Union{Nothing, Matrix{Float32}} # latent representation
    Z_cluster::Union{Nothing, Matrix{Float32}} 
    UMAP::Union{Nothing, Matrix{Float32}} 

    # Constructor to allow initializing Z as nothing
    function Autoencoder(; encoder::Union{Chain, Dense}, decoder::Union{Chain, Dense}, HP::Hyperparameter)
        new(encoder, decoder, HP, nothing, nothing, nothing)
    end
end
(AE::Autoencoder)(X) = AE.decoder(AE.encoder(X))
Flux.@functor Autoencoder

#---training function:
function train_AE!(X::AbstractMatrix{<:AbstractFloat}, AE::Autoencoder)

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

    return mean_trainlossPerEpoch
end