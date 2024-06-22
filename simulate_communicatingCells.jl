"""
simulate_interacting_singleCells(n_cells::Int = 1000, n_genes::Int = 60, n_groups::Int = 4;
        dropout_rate::Real = 0.1,
        HEG_mean::Real = 10,
        HEG_sd::Real = 2,
        communication_mean::Real = 10,
        communication_sd::Real = 1,
        technical_noise_mean::Real = 0,
        technical_noise_sd::Real = 4,
        n_noise_genes::Int = 10,
        seed::Int = 1234,
        communication_graph::Union{Matrix, Nothing} = nothing
    )

Simulates the expression levels of genes in interacting single cells. 

# Arguments
- `n_cells::Int`: The number of cells to simulate. Default is 1000.
- `n_genes::Int`: The number of genes to simulate. Default is 60 (is combined with `n_noise_genes`).
- `n_groups::Int`: The number of cell groups to simulate. Default is 4.
- `dropout_rate::Real`: The rate of dropout, i.e., the probability that an expression value is set to zero due to technical noise. Default is 0.2.
- `HEG_mean::Real`: The mean of the additional expression for high-expression genes. Default is 10.
- `HEG_sd::Real`: The standard deviation of the additional expression for high-expression genes. Default is 3.
- `communication_mean::Real`: The mean of the expression for communication genes. Default is 10.
- `communication_sd::Real`: The standard deviation of the expression for communication genes. Default is 3.
- `technical_noise_mean::Real`: The mean of the technical noise added to the expression values. Default is 0.
- `technical_noise_sd::Real`: The standard deviation of the technical noise added to the expression values. Default is 4.
- `n_noise_genes::Int`: The number of noise genes to add to the dataset. Default is 10.
- `seed::Int`: The seed for the random number generator. Default is 1234.
- `communication_graph::Union{Matrix, Nothing}`: The communication graph between cell groups. If `nothing`, a random communication graph will be generated. Default is `nothing`.

# Returns
- `dataset::Matrix{Float32}`: The simulated expression dataset.
- `group_communication_matrix::Matrix{Int}`: The communication graph between cell groups.
- `cell_group_assignments::Vector{String}`: The assignments of cells to cell groups.
- `receptor_genes::Dict{Any, Any}`: The receptor genes for each cell group.
- `sel_receptors::Vector{Any}`: The selected receptor genes for each cell group.
- `ligand_genes::Dict{Any, Any}`: The ligand genes for each cell group.
- `sel_ligands::Vector{Any}`: The selected ligand genes for each cell group.
"""
function simulate_interacting_singleCells(n_cells::Int = 1000, n_genes::Int = 60, n_groups::Int = 4;
        dropout_rate::Real = 0.1,
        HEG_mean::Real = 10,
        HEG_sd::Real = 2,
        communication_mean::Real = 10,
        communication_sd::Real = 1,
        technical_noise_mean::Real = 0,
        technical_noise_sd::Real = 4,
        n_noise_genes::Int = 10,
        seed::Int = 1234,
        communication_graph::Union{Matrix, Nothing} = nothing
    )

    Random.seed!(seed)

    # Check if the number of cells and genes are divisible by the number of groups
    if n_cells % n_groups != 0
        error("Number of cells must be divisible by the number of groups.")
    end
    if n_genes % n_groups != 0
        error("Number of genes must be divisible by the number of groups.")
    end


    # Create cell types
    cell_groups = ["Group " * string(i) for i in 1:n_groups]
    n_cells_per_group = n_cells ÷ length(cell_groups)
    n_genes_per_group = n_genes ÷ length(cell_groups)


    # Create a dictionary where each group in cell groups is a key to a number of genes
    highly_expr_genes = Dict()
    for (i, group) in enumerate(cell_groups)
        highly_expr_genes[group] = (i-1)*n_genes_per_group+1:(i-1)*n_genes_per_group + 2*n_genes_per_group÷length(cell_groups)
    end


    # Initialize the dataset
    dataset = zeros(n_cells, n_genes)
    cell_group_assignments = repeat(cell_groups, inner=n_cells_per_group)

    # Populate the dataset
    for (i, cell_group) in enumerate(cell_group_assignments)
        for gene in 1:n_genes
            # Set baseline expression for all genes
            expression = rand(Normal(technical_noise_mean, technical_noise_sd))  # Baseline expression
            
            # If the gene is highly expressed in this cell type, increase its expression
            if gene in highly_expr_genes[cell_group]
                expression += rand(Normal(HEG_mean, HEG_sd))  # Add additional expression for high-expression genes
            end
            
            dataset[i, gene] = max(expression, 0)  # Ensure expression is non-negative
        end
    end


    # Define a binary length(cell_groups)xlength(cell_groups) matrix, where each row sum is one 
    # (defines which cell group (row, ligand) communicates with which other cell group (column, receptor))
    if isnothing(communication_graph)
        group_communication_matrix = zeros(Int, length(cell_groups), length(cell_groups))
        for i in 1:length(cell_groups)
            group_communication_matrix[i, rand(1:length(cell_groups))] = 1
        end
    else
        group_communication_matrix = communication_graph
    end


    # Define a communication graph for the cell groups:
    ligand_genes = Dict()
    for (i, group) in enumerate(cell_groups)
        ligand_genes[group] = (i-1)*n_genes_per_group + 2*n_genes_per_group÷length(cell_groups) + 1:(i-1)*n_genes_per_group + 3*n_genes_per_group÷length(cell_groups)
    end

    sel_ligands = []
    for (i, cell_group) in enumerate(cell_groups)
        # Retrieve the communication genes for this cell group
        comm_genes = ligand_genes[cell_group]

        #randomly select one gene of comm_genes to be the ligand
        ligand = sample(comm_genes, 1, replace=false)
        push!(sel_ligands, ligand)

        #for the cells belonging to the cell group, randomly select half of them to express the ligand
        cell_indices = findall(x -> x == cell_group, cell_group_assignments)
        num_cells_per_group = length(cell_indices) #÷ 3  # Third the cells per group
        selected_cells = sample(cell_indices, num_cells_per_group, replace=false)
        dataset[selected_cells, comm_genes] .= rand(Normal(communication_mean, communication_sd), size(dataset[selected_cells, comm_genes]))
    end


    # Define a simple communication graph 
    receptor_genes = Dict()
    for (i, group) in enumerate(cell_groups)
        receptor_genes[group] = (i-1)*n_genes_per_group + 3*n_genes_per_group÷length(cell_groups) + 1:(i-1)*n_genes_per_group + 4*n_genes_per_group÷length(cell_groups)
    end

    sel_receptors = []
    for (i, cell_group) in enumerate(cell_groups)
        if sum(group_communication_matrix[:, i]) != 0
            # Retrieve the communication genes for this cell group
            comm_genes = receptor_genes[cell_group]

            #randomly select one gene of comm_genes to be the ligand
            receptor = sample(comm_genes, 1, replace=false)
            push!(sel_receptors, receptor)

            #for the cells belonging to the cell group, randomly select half of them to express the ligand
            cell_indices = findall(x -> x == cell_group, cell_group_assignments)
            num_cells_per_group = length(cell_indices) #÷ 3  # Third the cells per group
            selected_cells = sample(cell_indices, num_cells_per_group, replace=false)
            dataset[selected_cells, comm_genes] .= rand(Normal(communication_mean, communication_sd), size(dataset[selected_cells, comm_genes]))
        else
            push!(sel_receptors, [])
        end
    end


    #Add noise genes to the end of the data matrix (columns):
    dataset = hcat(dataset, rand(Normal(technical_noise_mean, technical_noise_sd), n_cells, n_noise_genes))


    # Add technical noise by randomly setting elements of the count matrix to zero:
    for i in 1:n_cells
        for j in 1:n_genes
            p = rand(Bernoulli(dropout_rate))
            if p == 1
                dataset[i, j] = 0
            end
        end
    end 

    # Again ensure that there are no negative elements in the data:
    dataset = max.(dataset, 0)

    # Round all Float values integers:
    dataset = round.(Int, dataset)

    return Float32.(dataset), group_communication_matrix, cell_group_assignments, receptor_genes, sel_receptors, ligand_genes, sel_ligands
end

### Test:
# Define which groups sends signals to which group in a binary communication graph/matrix (sender groups x receiver groups):
# Note: This is just an example.
# communication_graph = [1 0 0 0; 0 1 0 0; 0 1 0 0; 0 0 1 0];

# Generate the simulated data (set a seed for reproducibility):
# dataset, group_communication_matrix, cell_group_assignments, sel_receptors, sel_ligands = simulate_interacting_singleCells(1000, 60, 4; seed=7, communication_graph = communication_graph)