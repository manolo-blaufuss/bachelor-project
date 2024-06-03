function mse(X, Y, B1, B2, descr1, descr2)
    prediction1 = X * B1
    prediction2 = X * B2
    mse1 = mean((prediction1 - Y).^2, dims=1)
    mse2 = mean((prediction2 - Y).^2, dims=1)

    p = plot([mse1', mse2'], label = [descr1 descr2], title = "MSEs", xaxis = "Gene Index", yaxis = "MSE")
    display(p)
end