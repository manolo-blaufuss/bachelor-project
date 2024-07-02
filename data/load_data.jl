#------------------------------
# Setup: 
#------------------------------
#---Activate the enviroment:
using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

#if not loaded yet run:
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("DelimitedFiles")

using CSV;
using DataFrames;
using DelimitedFiles;

#depending on where the files are included the following might be redundant ...
projectpath = joinpath(@__DIR__, "../"); 
datapath = projectpath * "/data/"; 



#------------------------------
# Load data: 
#------------------------------
#---Load normalized count data (cells x genes):
X_source = readdlm(datapath * "X_source_norm.txt", '\t', Float32); #Count matrix of the original cells
X_target = readdlm(datapath * "X_target_norm.txt", '\t', Float32); #Count matrix of the target cells based on the initial matching using CellChat interaction scores

#---Load metadata:
Metadata_df = CSV.read(datapath * "Metadata_df.csv", DataFrame);

#---Load gene names:
genenames = vec(readdlm(datapath * "sel_genes.txt", '\t', String));
ligands = vec(readdlm(datapath * "Ligands.txt", '\t', String));
receptors = vec(readdlm(datapath * "Receptors.txt", '\t', String));
DEGs = vec(readdlm(datapath * "DEGs.txt", '\t', String));
noisegenes = vec(readdlm(datapath * "Noisegenes.txt", '\t', String));

#---Load the cell-cell interaction (CCI) tables determined via CellChat:
#1) Whole data:
CCI_all = CSV.read(datapath * "CCI_table.csv", DataFrame);
#2) Subset data (cells of 4 selected types):
CCI_sub = CSV.read(datapath * "CCI_table_sub.csv", DataFrame);
#3) Subset data (cells of 4 selected types), summed over ligands and receptor interactions:
CCI_sub_sum = CSV.read(datapath * "CCIsum_table_sub.csv", DataFrame);

#---Load the predefined cell-group interactions:
#The first column contains the source cell types, all types are considered as sender types. 
#The second column consists of the target cell types, which were determined by the top interacting cell group scores according to CellChat in CCI_sub_sum.
group_interactions = readdlm(datapath * "group_interactions.txt", '\t', String);