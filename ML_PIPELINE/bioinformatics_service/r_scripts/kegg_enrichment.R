# r_scripts/kegg_enrichment.R
# KEGG pathway enrichment analysis using clusterProfiler

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(enrichplot)
  library(ggplot2)
})

run_kegg_enrichment <- function(gene_list_path, organism, output_dir, pvalue_cutoff, qvalue_cutoff) {
  tryCatch({
    # Read gene list
    gene_df <- read.csv(gene_list_path, stringsAsFactors = FALSE)
    gene_list <- gene_df$gene_id
    
    cat(sprintf("KEGG enrichment for %d genes\n", length(gene_list)))
    cat(sprintf("Organism: %s\n", organism))
    
    # Select organism database
    if (organism == "mmu") {
      orgdb <- "org.Mm.eg.db"
      kegg_organism <- "mmu"
    } else if (organism == "hsa") {
      orgdb <- "org.Hs.eg.db"
      kegg_organism <- "hsa"
    } else {
      stop(paste("Unsupported organism:", organism))
    }
    
    library(orgdb, character.only = TRUE)
    
    # Convert gene IDs to Entrez
    cat("Converting gene IDs to Entrez...\n")
    gene_mapping <- bitr(
      gene_list,
      fromType = "ENSEMBL",
      toType = "ENTREZID",
      OrgDb = orgdb
    )
    
    # Save gene ID conversion
    conversion_file <- file.path(output_dir, "gene_id_conversion.csv")
    write.csv(gene_mapping, conversion_file, row.names = FALSE)
    cat(sprintf("Converted %d/%d genes\n", nrow(gene_mapping), length(gene_list)))
    
    entrez_genes <- gene_mapping$ENTREZID
    
    if (length(entrez_genes) == 0) {
      cat("No genes could be converted to Entrez IDs\n")
      return(0)
    }
    
    # Run KEGG enrichment
    cat("Running KEGG enrichment...\n")
    kegg_result <- enrichKEGG(
      gene = entrez_genes,
      organism = kegg_organism,
      pvalueCutoff = pvalue_cutoff,
      qvalueCutoff = qvalue_cutoff
    )
    
    if (is.null(kegg_result) || nrow(kegg_result@result) == 0) {
      cat("No enriched pathways found\n")
      return(0)
    }
    
    # Save results
    results_file <- file.path(output_dir, "kegg_enrichment_results.csv")
    write.csv(kegg_result@result, results_file, row.names = FALSE)
    
    num_pathways <- nrow(kegg_result@result)
    cat(sprintf("Found %d enriched pathways\n", num_pathways))
    
    # Create visualizations
    if (num_pathways > 0) {
      # Dotplot
      dotplot_file <- file.path(output_dir, "kegg_dotplot.png")
      png(dotplot_file, width = 800, height = 600)
      print(dotplot(kegg_result, showCategory = min(20, num_pathways)))
      dev.off()
      
      # Barplot
      barplot_file <- file.path(output_dir, "kegg_barplot.png")
      png(barplot_file, width = 800, height = 600)
      print(barplot(kegg_result, showCategory = min(20, num_pathways)))
      dev.off()
    }
    
    return(num_pathways)
    
  }, error = function(e) {
    cat(sprintf("Error in KEGG enrichment: %s\n", e$message))
    stop(e$message)
  })
}
