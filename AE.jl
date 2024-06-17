#--Hyperparameter:
mutable struct Hyperparameter
    zdim::Int # number of latent dimensions
    epochs::Int # number of training epochs
    batchsize::Int # batchsize for training
    η::Float32 # learning rate
    λ::Float32 # weight decay parameter

    # Constructor with default values
    function Hyperparameter(; zdim::Int=10, epochs::Int=50, batchsize::Int=2^7, η::Float32=0.01f0, λ::Float32=0.0f0)
        new(zdim, epochs, batchsize, η, λ)
    end 
end

#---AE architecture:
mutable struct Autoencoder
    encoder::Union{Chain, Dense} # encoder
    decoder::Union{Chain, Dense} # decoder
    HP::Hyperparameter # hyperparameters
    Z::Union{Nothing, Matrix{Float32}} # latent representation

    # Constructor to allow initializing Z as nothing
    function Autoencoder(; encoder::Union{Chain, Dense}, decoder::Union{Chain, Dense}, HP::Hyperparameter)
        new(encoder, decoder, HP, nothing)
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

#---training function vanilla AE:
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

#---help functions for VAE:
function divide_dimensions(x::AbstractArray{T}) where T
    n = size(x, 1)
    mid = div(n, 2)
    first_half = x[1:mid, :]
    second_half = x[mid+1:end, :]
    return first_half, second_half
end

# Reparameterization trick
function reparameterize(mu::AbstractArray{T}, logvar::AbstractArray{T}) where T
    sigma = exp.(0.5f0 .* logvar)
    epsilon = randn(Float32, size(mu))
    return mu .+ sigma .* epsilon
end

# Loss function
function vae_loss_gaußian(x::AbstractArray{T}, encoder::Union{Chain, Dense}, decoder::Union{Chain, Dense}; β::Float32=1.0f0) where T
    # encoder mu and logvar
    mu_enc, logvar_enc = divide_dimensions(encoder(x))

    # reparametrization trick
    z = reparameterize(mu_enc, logvar_enc)

    # decoder mu and logvar
    mu_dec, logvar_dec = divide_dimensions(decoder(z))

    # Reconstruction loss
    recon_loss = 0.5f0 * sum((x .- mu_dec).^2 ./ exp.(logvar_dec) .+ logvar_dec) 

    # KL divergence
    kl_loss = -0.5f0 * sum(1f0 .+ logvar_enc .- mu_enc.^2 .- exp.(logvar_enc))

    return recon_loss + β * kl_loss 
end

# Latent representation (expected)
function get_VAE_latentRepresentation(encoder::Union{Chain, Dense}, X::AbstractArray{T}; sampling::Bool=false) where T
    μ, logvar, z = nothing, nothing, nothing

    if sampling     
        μ, logvar = divide_dimensions(encoder(X))
        z = reparameterize(μ, logvar)
    else
        μ, logvar = divide_dimensions(encoder(X))
    end

    return μ, logvar, z
end

#---training function Variational AE:
function train_gaußianVAE!(X::AbstractMatrix{<:AbstractFloat}, AE::Autoencoder; β::Float32=1.0f0)

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
                vae_loss_gaußian(batch, enc, dec; β=β)
            end
            push!(batchLosses, batchLoss)

            Flux.update!(opt_state, (AE.encoder, AE.decoder), (grads[1], grads[2]))
        end

        push!(mean_trainlossPerEpoch, mean(batchLosses))
    end

    AE.Z = get_VAE_latentRepresentation(AE.encoder, X)[1]

    return mean_trainlossPerEpoch
end