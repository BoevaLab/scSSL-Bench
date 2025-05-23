
# Author : Kok Siong Ang 
# Date : 17/09/2019
# Proj : Run MNN Correct pipeline

########################
#load packages
rm(list=ls())

library(Seurat)  # Seurat 2 version
library(batchelor)
library(Rtsne)
library(scales)
library(glue)


# setting directory
# setwd('/home/yxh/gitrepo/clMining/scmoco/moco')
source('find_anchors_utils.R')

########################
#settings

filter_genes = F
filter_cells = F
normData = T
Datascaling = T
regressUMI = F
min_cells = 0
min_genes = 0
norm_method = "LogNormalize"
scale_factor = 10000

numVG = 2000
npcs = 30
batch_label = "batchlb"
celltype_label = "CellType"


########################
# MouseCellAtlas Dataset
#+++++++++++++++++++++++
dname = "dataset2"
read_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Data/{dname}/")
expr_filename = 'filtered_total_batch1_seqwell_batch2_10x.txt'
metadata_filename = 'filtered_total_sample_ext_organ_celltype_batch.txt'

# reading
expr_mat = read.table(file = paste0(read_dir,expr_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata = read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'ct'] = 'CellType'
expr_mat <- expr_mat[, rownames(metadata)]



########################
#  PBMC Dataset
#+++++++++++++++++++++++
dname = "dataset5"
read_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Data/{dname}/")

b1_exprs_filename = "b1_exprs.txt"
b2_exprs_filename = "b2_exprs.txt"
b1_celltype_filename = "b1_celltype.txt"
b2_celltype_filename = "b2_celltype.txt"

batch_label = "batchlb"
celltype_label = "CellType"

# read data 

b1_exprs <- read.table(file = paste0(read_dir,b1_exprs_filename),sep="\t",header=T,row.names=1,check.names = F)
b2_exprs <- read.table(file = paste0(read_dir,b2_exprs_filename),sep="\t",header=T,row.names=1,check.names = F)
b1_celltype <- read.table(file = paste0(read_dir,b1_celltype_filename),sep="\t",header=T,row.names=1,check.names = F)
b2_celltype <- read.table(file = paste0(read_dir,b2_celltype_filename),sep="\t",header=T,row.names=1,check.names = F)

# b1_celltype$cell <- rownames(b1_celltype)
b1_celltype <- b1_celltype[colnames(b1_exprs),]
# b2_celltype$cell <- rownames(b2_celltype)
b2_celltype <- b2_celltype[colnames(b2_exprs),]

b1_metadata <- as.data.frame(b1_celltype)
b2_metadata <- as.data.frame(b2_celltype)

b1_metadata$batchlb <- 'Batch1'
b2_metadata$batchlb <- 'Batch2'

expr_mat = cbind(b1_exprs,b2_exprs)
metadata = rbind(b1_metadata, b2_metadata)

expr_mat <- expr_mat[, rownames(metadata)]


########################
#  Pancreas Dataset
#+++++++++++++++++++++++
dname = "dataset4"
read_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Data/{dname}/")

expr_mat_filename = "myData_pancreatic_5batches.txt"
metadata_filename = "mySample_pancreatic_5batches.txt"

batch_label = "batchlb"
celltype_label = "CellType"

# read data 
expr_mat <- read.table(file = paste0(read_dir,expr_mat_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata <- read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'celltype'] <- "CellType"

expr_mat <- expr_mat[, rownames(metadata)]


##############################################   Shared Operations  ##############################################
anchors_return = find_anchors(expr_mat, metadata, 
                          filter_genes = filter_genes, filter_cells = filter_cells,
                          normData = normData, Datascaling = Datascaling, regressUMI = regressUMI, 
                          min_cells = min_cells, min_genes = min_genes, 
                          norm_method = norm_method, scale_factor = scale_factor, 
                          numVG = numVG, npcs = npcs, 
                          batch_label = batch_label, celltype_label = celltype_label,
                          )


anchors = anchors_return$anchors
bso.list = anchors_return$Xlist

# head(bso.list[[1]]@meta.data)

df_anchor = anchors@anchors  
df_anchor$name1 = 'empty'
df_anchor$name2 = 'empty'

dim(df_anchor)

for (i in 1:dim(df_anchor)[1]){  # visit all MNN pairs, here we suppose that every cell has an unique name 
    c1 = df_anchor[i, 'cell1']
    c2 = df_anchor[i, 'cell2']
    d1 = df_anchor[i, 'dataset1']
    d2 = df_anchor[i, 'dataset2']
    df_anchor[i, 'name1'] = colnames(bso.list[[d1]])[c1]
    df_anchor[i, 'name2'] = colnames(bso.list[[d2]])[c2]
}


write.table(df_anchor, file=paste0(read_dir, glue("seuratAnchors.csv")), quote=F, sep=',', row.names = T, col.names = NA)
