#---Plot a heatmap, e.g., of the latent representation of an autoencoder:
heatmap(X,
    xlabel = "X-axis Label",
    ylabel = "Y-axis Label",
    title = "Heatmap Title",
    color = :vik #choose from:"https://docs.juliaplots.org/latest/generated/colorschemes/"
)


#---Plot the mean trainloss per epoch:
loss_plot = plot(1:length(mean_trainlossPerEpoch), mean_trainlossPerEpoch,
     title = "Mean train loss per epoch",
     xlabel = "Epoch",
     ylabel = "Loss",
     legend = true,
     label = "Train loss",
     linecolor = :red,
     linewidth = 2
)

#---Plot a heatmap of the absolute values of pearson correlation coefficients between latent dimensions:
using StatsBase; #I think this package is required for the correlation function cor()
heatmap(abs.(cor(Z, dims=2)), #depending on size(Z), dims=1 or dims=2
    xlabel = "X-axis Label",
    ylabel = "Y-axis Label",
    title = "Heatmap Title",
    color = :vik #choose from:"https://docs.juliaplots.org/latest/generated/colorschemes/"
)