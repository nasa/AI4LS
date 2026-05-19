# r_scripts/deseq2_wrapper.R
# DESeq2 differential expression analysis wrapper

suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
})

run_deseq2 <- function(count_matrix_path, condition_column, control_group, treatment_group, 
                       output_dir, padj_threshold, log2fc_threshold) {
  tryCatch({
    # Read count matrix
    count_data <- read.csv(count_matrix_path, row.names = 1, check.names = FALSE)
    
    cat(sprintf("Count matrix: %d genes x %d samples\n", nrow(count_data), ncol(count_data)))
    
    # Extract condition from column names (assumes last column or specific pattern)
    # This is a simplified version - adjust based on your metadata structure
    conditions <- rep(control_group, ncol(count_data))
    # Mark treatment samples (this logic may need adjustment)
    treatment_indices <- grep(treatment_group, colnames(count_data), ignore.case = TRUE)
    conditions[treatment_indices] <- treatment_group
    
    # Create metadata
    col_data <- data.frame(
      condition = factor(conditions, levels = c(control_group, treatment_group)),
      row.names = colnames(count_data)
    )
    
    cat(sprintf("Samples: %d control, %d treatment\n", 
                sum(col_data$condition == control_group),
                sum(col_data$condition == treatment_group)))
    
    # Create DESeq2 object
    dds <- DESeqDataSetFromMatrix(
      countData = round(count_data),
      colData = col_data,
      design = ~ condition
    )
    
    # Filter low count genes
    keep <- rowSums(counts(dds)) >= 10
    dds <- dds[keep,]
    
    cat(sprintf("After filtering: %d genes\n", nrow(dds)))
    
    # Run DESeq2
    cat("Running DESeq2...\n")
    dds <- DESeq(dds)
    
    # Get results
    res <- results(dds, contrast = c("condition", treatment_group, control_group))
    
    # Order by adjusted p-value
    res_ordered <- res[order(res$padj),]
    
    # Save all results
    results_file <- file.path(output_dir, "deseq2_all_results.csv")
    write.csv(as.data.frame(res_ordered), results_file)
    
    # Count significant genes
    sig_genes <- sum(res$padj < padj_threshold & abs(res$log2FoldChange) > log2fc_threshold, na.rm = TRUE)
    up_genes <- sum(res$padj < padj_threshold & res$log2FoldChange > log2fc_threshold, na.rm = TRUE)
    down_genes <- sum(res$padj < padj_threshold & res$log2FoldChange < -log2fc_threshold, na.rm = TRUE)
    
    cat(sprintf("Significant genes: %d (up: %d, down: %d)\n", sig_genes, up_genes, down_genes))
    
    # Create volcano plot
    volcano_file <- file.path(output_dir, "volcano_plot.png")
    png(volcano_file, width = 800, height = 600)
    plot(res$log2FoldChange, -log10(res$pvalue),
         xlab = "log2 Fold Change", ylab = "-log10(p-value)",
         main = "Volcano Plot",
         pch = 20, col = ifelse(res$padj < padj_threshold, "red", "gray"))
    abline(h = -log10(0.05), col = "blue", lty = 2)
    abline(v = c(-log2fc_threshold, log2fc_threshold), col = "blue", lty = 2)
    dev.off()
    
    # Create MA plot
    ma_file <- file.path(output_dir, "ma_plot.png")
    png(ma_file, width = 800, height = 600)
    plotMA(res, main = "MA Plot", ylim = c(-5, 5))
    dev.off()
    
    return(list(
      num_genes = nrow(res),
      num_significant = sig_genes,
      num_upregulated = up_genes,
      num_downregulated = down_genes
    ))
    
  }, error = function(e) {
    cat(sprintf("Error in DESeq2: %s\n", e$message))
    stop(e$message)
  })
}
