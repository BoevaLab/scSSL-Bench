library(Seurat)  # Seurat >= 3


find_anchors <- function(expr_mat_mnn, metadata, 
                            filter_genes = T, filter_cells = T,
                            normData = T, Datascaling = T, regressUMI = F, 
                            min_cells = 10, min_genes = 300, norm_method = "LogNormalize", scale_factor = 10000, 
                            # b_x_low_cutoff = 0.0125, b_x_high_cutoff = 3, b_y_cutoff = 0.5, 
                            numVG = 300, npcs = 30,
                            batch_label = "batchlb", celltype_label = "CellType",
                            outfilename_prefix='simLinear_10_batches')
{

  ##########################################################
  # preprocessing

  if(filter_genes == F) {
    min_cells = 0
  }
  if(filter_cells == F) {
    min_genes = 0
  }

  b_seurat <- CreateSeuratObject(counts = expr_mat_mnn, meta.data = metadata, project = "seurat_benchmark", 
                                 min.cells = min_cells, min.genes = min_genes)

  # split object by batch
  bso.list <- SplitObject(b_seurat, split.by = batch_label)

  # normalize and identify variable features for each dataset independently
  bso.list <- lapply(X = bso.list, FUN = function(x) {
      x <- NormalizeData(x)
      x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = numVG)
  })

  # take the share hvg features, and subsets
  features <- SelectIntegrationFeatures(object.list = bso.list)

  bso.anchors <- FindIntegrationAnchors(object.list = bso.list, anchor.features = features)

  #==================TO-DO===================
  # transform the dataset.id in bso.anchors to 'dataset.name'

  return (list(anchors=bso.anchors, Xlist=bso.list))
}