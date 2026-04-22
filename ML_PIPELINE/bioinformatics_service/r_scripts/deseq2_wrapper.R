# bioinformatics_service/r_scripts/deseq2_wrapper.R

run_deseq2 <- function(count_matrix_path, metadata_path, condition_column, 
                       control_group, treatment_group, output_dir,
                       padj_threshold, log2fc_threshold) {
  
  library(DESeq2)
  library(ggplot2)

  cat("padj_threshold in run_deseq2:", padj_threshold, "\n")
  cat("log2fc_threshold in run_deseq2:", log2fc_threshold, "\n")
  
  # Load data
  counts <- read.csv(count_matrix_path, row.names = 1, check.names = FALSE)
  metadata <- read.csv(metadata_path, row.names = 1, check.names = FALSE)
  
  # Ensure counts are integer
  counts <- round(counts)
  
  # Clean column name for formula
  clean_condition_column <- gsub(" ", "_", condition_column)
  clean_condition_column <- gsub("\\[", "_", clean_condition_column)
  clean_condition_column <- gsub("\\]", "_", clean_condition_column)
  clean_condition_column <- gsub("__+", "_", clean_condition_column)
  
  cat("Original condition column:", condition_column, "\n")
  cat("Cleaned condition column:", clean_condition_column, "\n")
  
  # Rename column in metadata if needed
  if (condition_column != clean_condition_column) {
    colnames(metadata)[colnames(metadata) == condition_column] <- clean_condition_column
  }
  
  # Convert to factor
  cat("Converting condition column to factor...\n")
  cat("Unique values before factor:", unique(metadata[[clean_condition_column]]), "\n")
  
  metadata[[clean_condition_column]] <- as.factor(metadata[[clean_condition_column]])
  factor_levels <- levels(metadata[[clean_condition_column]])
  
  cat("Factor levels:", paste(factor_levels, collapse=", "), "\n")
  cat("Requested control_group:", control_group, "\n")
  cat("Requested treatment_group:", treatment_group, "\n")
  
  # Find matching level for control group (case-insensitive, partial match)
  control_match <- factor_levels[grep(control_group, factor_levels, ignore.case = TRUE)]
  treatment_match <- factor_levels[grep(treatment_group, factor_levels, ignore.case = TRUE)]
  
  # If exact match not found, try first character match or use first/last levels
  if (length(control_match) == 0) {
    cat("WARNING: Control group not found, trying alternatives...\n")
    # Try case-insensitive exact match
    control_match <- factor_levels[tolower(factor_levels) == tolower(control_group)]
    if (length(control_match) == 0) {
      cat("Using first level as control:", factor_levels[1], "\n")
      control_match <- factor_levels[1]
    }
  } else {
    control_match <- control_match[1]  # Take first match
  }
  
  if (length(treatment_match) == 0) {
    cat("WARNING: Treatment group not found, trying alternatives...\n")
    treatment_match <- factor_levels[tolower(factor_levels) == tolower(treatment_group)]
    if (length(treatment_match) == 0) {
      # Use the level that's not control
      treatment_match <- factor_levels[factor_levels != control_match][1]
      cat("Using alternative treatment:", treatment_match, "\n")
    }
  } else {
    treatment_match <- treatment_match[1]
  }
  
  cat("Actual control_group:", control_match, "\n")
  cat("Actual treatment_group:", treatment_match, "\n")
  
  # Create design formula
  design_formula <- as.formula(paste0("~ ", clean_condition_column))
  
  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = metadata,
    design = design_formula
  )
  
  # Set reference level with matched control group
  cat("Setting reference level...\n")
  dds[[clean_condition_column]] <- relevel(dds[[clean_condition_column]], ref = control_match)
  
  # Run DESeq2
  cat("Running DESeq2...\n")
  dds <- DESeq(dds)
  
  # Get results with matched groups
  res <- results(dds, contrast = c(clean_condition_column, treatment_match, control_match))
  
  # Order by adjusted p-value
  res_ordered <- res[order(res$padj), ]
  
  # Filter significant genes
  sig_genes <- subset(res_ordered, padj < padj_threshold & abs(log2FoldChange) > log2fc_threshold)
  
  # Save results
  write.csv(as.data.frame(res_ordered), file.path(output_dir, "deseq2_all_results.csv"))
  write.csv(as.data.frame(sig_genes), file.path(output_dir, "deseq2_significant_genes.csv"))
  
  # Volcano plot
  volcano_data <- as.data.frame(res_ordered)
  volcano_data$significant <- ifelse(
    volcano_data$padj < padj_threshold & abs(volcano_data$log2FoldChange) > log2fc_threshold,
    "Significant", "Not Significant"
  )
  
  p_volcano <- ggplot(volcano_data, aes(x = log2FoldChange, y = -log10(padj), color = significant)) +
    geom_point(alpha = 0.6, size = 1.5) +
    scale_color_manual(values = c("gray", "red")) +
    geom_vline(xintercept = c(-log2fc_threshold, log2fc_threshold), linetype = "dashed") +
    geom_hline(yintercept = -log10(padj_threshold), linetype = "dashed") +
    theme_minimal() +
    labs(title = paste0(treatment_match, " vs ", control_match),
         x = "Log2 Fold Change",
         y = "-Log10 Adjusted P-value")
  
  ggsave(file.path(output_dir, "volcano_plot.png"), p_volcano, width = 10, height = 8)
  
  # MA plot
  png(file.path(output_dir, "ma_plot.png"), width = 800, height = 600)
  DESeq2::plotMA(res, main = paste0(treatment_match, " vs ", control_match))
  dev.off()
  
  # Count statistics
  num_genes <- nrow(res_ordered)
  num_upregulated <- sum(sig_genes$log2FoldChange > 0, na.rm = TRUE)
  num_downregulated <- sum(sig_genes$log2FoldChange < 0, na.rm = TRUE)
  num_significant <- nrow(sig_genes)
  
  cat("Analysis complete!\n")
  cat("Total genes:", num_genes, "\n")
  cat("Significant genes:", num_significant, "\n")
  cat("Upregulated:", num_upregulated, "\n")
  cat("Downregulated:", num_downregulated, "\n")
  
  # Return as a named list
  result_list <- list(
    num_genes = as.integer(num_genes),
    num_upregulated = as.integer(num_upregulated),
    num_downregulated = as.integer(num_downregulated),
    num_significant = as.integer(num_significant)
  )
  
  return(result_list)
}
