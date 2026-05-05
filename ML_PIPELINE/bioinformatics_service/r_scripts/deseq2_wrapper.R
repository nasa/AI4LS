# bioinformatics_service/r_scripts/deseq2_wrapper.R

run_deseq2 <- function(count_matrix_path, metadata_path, condition_column, 
                       control_group, treatment_group, output_dir,
                       padj_threshold = 0.05, log2fc_threshold = 1.0) {
  
  library(DESeq2)
  library(ggplot2)
  
  # Load data
  counts <- read.csv(count_matrix_path, row.names = 1, check.names = FALSE)
  metadata <- read.csv(metadata_path, row.names = 1, check.names = FALSE)
  
  cat("Initial counts shape:", nrow(counts), "genes x", ncol(counts), "samples\n")
  cat("Metadata shape:", nrow(metadata), "samples x", ncol(metadata), "columns\n")
  
  # Check sample names match
  cat("Count sample names (first 5):", head(colnames(counts), 5), "\n")
  cat("Metadata sample names (first 5):", head(rownames(metadata), 5), "\n")
  
  # Ensure metadata and counts have matching samples
  common_samples <- intersect(colnames(counts), rownames(metadata))
  cat("Common samples:", length(common_samples), "\n")
  
  if (length(common_samples) == 0) {
    stop("No common samples between counts and metadata!")
  }
  
  # Subset to common samples
  counts <- counts[, common_samples]
  metadata <- metadata[common_samples, , drop = FALSE]
  
  cat("After matching - counts:", ncol(counts), "samples, metadata:", nrow(metadata), "samples\n")
  
  # Ensure counts are integer
  counts <- round(counts)
  
  # Filter out genes with all zeros or very low counts
  cat("Filtering low-count genes...\n")
  keep <- rowSums(counts) >= 10
  counts <- counts[keep, ]
  cat("After filtering:", nrow(counts), "genes remaining\n")
  
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
  
  # Check if column exists
  if (!(clean_condition_column %in% colnames(metadata))) {
    stop(paste("Column", clean_condition_column, "not found in metadata. Available columns:", 
               paste(colnames(metadata), collapse=", ")))
  }
  
  # Convert to factor
  cat("Converting condition column to factor...\n")
  cat("Values before conversion:\n")
  print(table(metadata[[clean_condition_column]]))
  
  metadata[[clean_condition_column]] <- as.factor(metadata[[clean_condition_column]])
  factor_levels <- levels(metadata[[clean_condition_column]])

  cat("conditions in deseq2: ")
  print(metadata[[clean_condition_column]])
  
  
  cat("Factor levels:", paste(factor_levels, collapse=", "), "\n")
  cat("Number of levels:", length(factor_levels), "\n")
  
  if (length(factor_levels) < 2) {
    stop("Need at least 2 factor levels for comparison")
  }
  
  # Find matching level for control group
  control_match <- NULL
  treatment_match <- NULL
  
  # Try exact match first
  if (control_group %in% factor_levels) {
    control_match <- control_group
    cat("Exact match for control:", control_match, "\n")
  } else {
    # Try case-insensitive match
    idx <- which(tolower(factor_levels) == tolower(control_group))
    if (length(idx) > 0) {
      control_match <- factor_levels[idx[1]]
      cat("Case-insensitive match for control:", control_match, "\n")
    } else {
      # Try grep (partial match)
      idx <- grep(control_group, factor_levels, ignore.case = TRUE)
      if (length(idx) > 0) {
        control_match <- factor_levels[idx[1]]
        cat("Partial match for control:", control_match, "\n")
      } else {
        # Default to first level
        control_match <- factor_levels[1]
        cat("WARNING: No match found for control '", control_group, "', using first level:", control_match, "\n")
      }
    }
  }
  
  # Try exact match for treatment
  if (treatment_group %in% factor_levels) {
    treatment_match <- treatment_group
    cat("Exact match for treatment:", treatment_match, "\n")
  } else {
    # Try case-insensitive match
    idx <- which(tolower(factor_levels) == tolower(treatment_group))
    if (length(idx) > 0) {
      treatment_match <- factor_levels[idx[1]]
      cat("Case-insensitive match for treatment:", treatment_match, "\n")
    } else {
      # Try grep (partial match)
      idx <- grep(treatment_group, factor_levels, ignore.case = TRUE)
      if (length(idx) > 0) {
        treatment_match <- factor_levels[idx[1]]
        cat("Partial match for treatment:", treatment_match, "\n")
      } else {
        # Default to second level
        other_levels <- factor_levels[factor_levels != control_match]
        if (length(other_levels) > 0) {
          treatment_match <- other_levels[1]
        } else {
          treatment_match <- factor_levels[2]
        }
        cat("WARNING: No match found for treatment '", treatment_group, "', using:", treatment_match, "\n")
      }
    }
  }
  
  # Validate matches are not NULL or NA
  cat("Validating matches...\n")
  cat("control_match:", control_match, "is.null:", is.null(control_match), "is.na:", is.na(control_match), "\n")
  cat("treatment_match:", treatment_match, "is.null:", is.null(treatment_match), "is.na:", is.na(treatment_match), "\n")
  
  if (is.null(control_match) || length(control_match) == 0) {
    stop(paste("Control match is NULL. Available levels:", paste(factor_levels, collapse=", ")))
  }
  
  if (is.na(control_match)) {
    stop(paste("Control match is NA. Available levels:", paste(factor_levels, collapse=", ")))
  }
  
  if (!(control_match %in% factor_levels)) {
    stop(paste("Control match", control_match, "not in factor levels:", paste(factor_levels, collapse=", ")))
  }
  
  if (is.null(treatment_match) || length(treatment_match) == 0) {
    stop(paste("Treatment match is NULL. Available levels:", paste(factor_levels, collapse=", ")))
  }
  
  if (is.na(treatment_match)) {
    stop(paste("Treatment match is NA. Available levels:", paste(factor_levels, collapse=", ")))
  }
  
  if (!(treatment_match %in% factor_levels)) {
    stop(paste("Treatment match", treatment_match, "not in factor levels:", paste(factor_levels, collapse=", ")))
  }
  
  if (control_match == treatment_match) {
    stop(paste("Control and treatment groups are the same:", control_match))
  }
  
  cat("Final validation passed!\n")
  cat("Using control:", control_match, "\n")
  cat("Using treatment:", treatment_match, "\n")
  
  # Create design formula
  design_formula <- as.formula(paste0("~ ", clean_condition_column))
  cat("Design formula:", as.character(design_formula), "\n")
  
  # Create DESeq2 dataset
  cat("Creating DESeqDataSet...\n")
  dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = metadata,
    design = design_formula
  )
  
  cat("DESeqDataSet created successfully\n")
  cat("Condition factor in dds:\n")
  print(table(dds[[clean_condition_column]]))
  
  # Set reference level
  cat("Setting reference level to:", control_match, "\n")
  cat("Current levels:", levels(dds[[clean_condition_column]]), "\n")
  
  dds[[clean_condition_column]] <- relevel(dds[[clean_condition_column]], ref = control_match)
  
  cat("Reference level set successfully\n")
  cat("New levels:", levels(dds[[clean_condition_column]]), "\n")
  
  # Run DESeq2
  cat("Running DESeq2 analysis...\n")
  dds <- tryCatch({
    DESeq(dds)
  }, error = function(e) {
    if (grepl("every gene contains at least one zero", e$message)) {
      cat("Standard normalization failed, using alternative method...\n")
      # Use alternative size factor estimation
      dds <- estimateSizeFactors(dds, type = "poscounts")
      dds <- estimateDispersions(dds)
      dds <- nbinomWaldTest(dds)
      return(dds)  # ← CRITICAL: Must return dds here
    } else {
      stop(e)
    }
  }) 

  cat("DESeq2 analysis complete\n")
  
  # Get results
  cat("Extracting results...\n")
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
  cat("Total genes analyzed:", num_genes, "\n")
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
