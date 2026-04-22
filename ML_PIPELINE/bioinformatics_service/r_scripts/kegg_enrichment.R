# bioinformatics_service/r_scripts/kegg_enrichment.R

run_kegg_enrichment <- function(gene_list_path, organism, output_dir,
                                pvalue_cutoff = 0.05, qvalue_cutoff = 0.2) {
  
  library(clusterProfiler)
  library(enrichplot)
  library(ggplot2)
  
  # Load organism database
  if (organism == "mmu") {
    library(org.Mm.eg.db)
    orgdb <- org.Mm.eg.db
    cat("Using mouse organism database (org.Mm.eg.db)\n")
  } else if (organism == "hsa") {
    library(org.Hs.eg.db)
    orgdb <- org.Hs.eg.db
    cat("Using human organism database (org.Hs.eg.db)\n")
  } else {
    stop(paste("Unsupported organism:", organism))
  }
  
  # Load gene list
  genes <- read.csv(gene_list_path, header = TRUE)
  gene_ids <- genes$gene_id
  
  cat("Input gene IDs (first 10):\n")
  print(head(gene_ids, 10))
  cat("Total genes:", length(gene_ids), "\n")
  
  # Convert ENSEMBL IDs to Entrez IDs
  cat("\nConverting ENSEMBL IDs to Entrez IDs...\n")
  
  # Try different ID types in case ENSEMBL doesn't work
  conversion_result <- tryCatch({
    bitr(gene_ids, 
         fromType = "ENSEMBL",
         toType = "ENTREZID",
         OrgDb = orgdb)
  }, error = function(e) {
    cat("ENSEMBL conversion failed, trying SYMBOL...\n")
    tryCatch({
      bitr(gene_ids, 
           fromType = "SYMBOL",
           toType = "ENTREZID",
           OrgDb = orgdb)
    }, error = function(e2) {
      cat("SYMBOL conversion failed, trying direct use...\n")
      # If gene_ids are already Entrez IDs
      data.frame(ENTREZID = gene_ids)
    })
  })
  
  if (nrow(conversion_result) == 0) {
    cat("ERROR: No genes could be converted\n")
    return(0)
  }
  
  entrez_ids <- conversion_result$ENTREZID
  
  cat("Converted genes:", length(entrez_ids), "\n")
  cat("Conversion rate:", round(length(entrez_ids) / length(gene_ids) * 100, 1), "%\n")
  cat("Entrez IDs (first 10):\n")
  print(head(entrez_ids, 10))
  
  # Save conversion mapping
  write.csv(conversion_result, 
            file.path(output_dir, "gene_id_conversion.csv"),
            row.names = FALSE)
  
  # Run KEGG enrichment
  cat("\nRunning KEGG enrichment analysis...\n")
  kegg_enrich <- enrichKEGG(
    gene = entrez_ids,
    organism = organism,
    pvalueCutoff = pvalue_cutoff,
    qvalueCutoff = qvalue_cutoff,
    pAdjustMethod = "BH"
  )
  
  # Save results
  if (!is.null(kegg_enrich) && nrow(kegg_enrich@result) > 0) {
    cat("Found", nrow(kegg_enrich@result), "enriched pathways\n")
    
    write.csv(as.data.frame(kegg_enrich), 
              file.path(output_dir, "kegg_enrichment_results.csv"),
              row.names = FALSE)
    
    # Dotplot
    if (nrow(kegg_enrich@result) > 0) {
      p_dot <- dotplot(kegg_enrich, showCategory = min(20, nrow(kegg_enrich@result))) +
        ggtitle("KEGG Pathway Enrichment")
      ggsave(file.path(output_dir, "kegg_dotplot.png"), p_dot, width = 12, height = 10)
      
      # Barplot
      p_bar <- barplot(kegg_enrich, showCategory = min(20, nrow(kegg_enrich@result))) +
        ggtitle("KEGG Pathway Enrichment")
      ggsave(file.path(output_dir, "kegg_barplot.png"), p_bar, width = 12, height = 10)
    }
    
    return(nrow(kegg_enrich@result))
  } else {
    cat("No enriched pathways found\n")
    return(0)
  }
}
