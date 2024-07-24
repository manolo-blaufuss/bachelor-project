# This script contains code provided by Niklas Brunn

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

function vae_loss_gaußian_fixedvariance(x::AbstractArray{T}, encoder::Union{Chain, Dense}, decoder::Union{Chain, Dense}; β::Float32=1.0f0, var::Float32=0.1f0) where T
    # encoder mu and logvar
    mu_enc, logvar_enc = divide_dimensions(encoder(x))

    # reparametrization trick
    z = reparameterize(mu_enc, logvar_enc)

    # decoder mu and logvar
    mu_dec = decoder(z)

    # Reconstruction loss
    recon_loss = (0.5f0 / var) * sum((x .- mu_dec).^2) + 0.5f0 * size(x, 2) * log(2f0 * Float32(π) * var)

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

function train_gaußianVAE_fixedvariance!(X::AbstractMatrix{<:AbstractFloat}, AE::Autoencoder; β::Float32=1.0f0, var::Float32=1.0f0)

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
                vae_loss_gaußian_fixedvariance(batch, enc, dec; β=β, var=var)
            end
            push!(batchLosses, batchLoss)

            Flux.update!(opt_state, (AE.encoder, AE.decoder), (grads[1], grads[2]))
        end

        push!(mean_trainlossPerEpoch, mean(batchLosses))
    end

    AE.Z = get_VAE_latentRepresentation(AE.encoder, X)[1]

    return mean_trainlossPerEpoch
end


function dimension_reduction(X::Matrix{Float32}, Y::Matrix{Float32}, method::String, latent_dim::Int; seed=42, save_figures=false)
    HP = Hyperparameter(zdim=latent_dim, epochs=20, batchsize=2^7, η=0.01f0, λ=0.0f0)
    akt = tanh_fast #relu, sigmoid, tanh_fast, tanh, ...
    if method == "VAE"
        modelseed = seed;
        Random.seed!(modelseed)
        # VAE architecture, default: 2 layers, akt
        encoder = Chain(Dense(size(X, 2), 2*HP.zdim, akt), Dense(2*HP.zdim, 2*HP.zdim))
        decoder = Chain(Dense(HP.zdim, 2* size(X, 2), akt), Dense(2 * size(X, 2), 2 * size(X, 2)))
        AE = Autoencoder(;encoder, decoder, HP)
        summary(AE)
        mean_trainlossPerEpoch = train_gaußianVAE!(X', AE)
        Z_X = get_VAE_latentRepresentation(AE.encoder, X'; sampling=false)[1]
        Z_Y = get_VAE_latentRepresentation(AE.encoder, Y'; sampling=false)[1]
    elseif method == "VAE_fixed"
        modelseed = seed;
        Random.seed!(modelseed)
        # VAE architecture, default: 2 layers, akt
        encoder = Chain(Dense(size(X, 2), 2*HP.zdim, akt), Dense(2*HP.zdim, 2*HP.zdim))
        decoder = Chain(Dense(HP.zdim, size(X, 2), akt), Dense(size(X, 2), size(X, 2)))
        AE = Autoencoder(;encoder, decoder, HP)
        summary(AE)
        mean_trainlossPerEpoch = train_gaußianVAE_fixedvariance!(X', AE)
        Z_X = get_VAE_latentRepresentation(AE.encoder, X'; sampling=false)[1]
        Z_Y = get_VAE_latentRepresentation(AE.encoder, Y'; sampling=false)[1]
    end
    if save_figures
        h1 = heatmap(Z_X', title="Latent Representation of X", xlabel="Latent Dimensions", ylabel="Observation Units")
        h2 = heatmap(Z_Y', title="Latent Representation of Y", xlabel="Latent Dimensions", ylabel="Observation Units")
        h3 = heatmap(abs.(cor(Z_X, dims=2)), xlabel = "Latent Dimensions", ylabel = "Latent Dimensions", title = "Absolute Correlation", color = :reds)
        p = plot(1:length(mean_trainlossPerEpoch), mean_trainlossPerEpoch, title = "Mean train loss per epoch", xlabel = "Epoch", ylabel = "Loss", legend = true, label = "Train loss", linecolor = :red, linewidth = 2)
        savefig(h1, "output/auto_output/Z_X.png")
        savefig(h1, "output/auto_output/Z_X.svg")
        savefig(h2, "output/auto_output/Z_Y.png")
        savefig(h2, "output/auto_output/Z_Y.svg")
        savefig(h3, "output/auto_output/correlation_latent_dims.png")
        savefig(h3, "output/auto_output/correlation_latent_dims.svg")
        savefig(p, "output/auto_output/train_loss.png")
        savefig(p, "output/auto_output/train_loss.svg")
    end
    return Z_X, Z_Y
end