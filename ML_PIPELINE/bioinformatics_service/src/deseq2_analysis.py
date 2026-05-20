# src/deseq2_analysis.py

import logging
import pandas as pd
from pathlib import Path
from typing import Dict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DESeq2Analysis:
    """DESeq2 differential expression analysis using R"""
    
    def __init__(self, output_base_path: str = "/app/results"):
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        # Source R script once during initialization
        r_script_path = Path("/app/r_scripts/deseq2_wrapper.R")
        if not r_script_path.exists():
            raise FileNotFoundError(f"R script not found: {r_script_path}")
        
        ro.r(f'source("{r_script_path}")')
        logger.info(f"Sourced R script: {r_script_path}")
    
    def run_analysis(
        self,
        dataset_id: str,
        condition_column: str,
        control_group: str,
        treatment_group: str,
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0
    ) -> Dict:
        """Run DESeq2 differential expression analysis"""
        try:
            logger.info(f"Running DESeq2 analysis")
            logger.info(f"  Dataset: {dataset_id}")
            logger.info(f"  Comparison: {treatment_group} vs {control_group}")
            logger.info(f"  using padj: {padj_threshold} and l2fc: {log2fc_threshold}")
            
            # Create output directory
            analysis_id = f"deseq2_{dataset_id}"
            output_dir = self.output_base_path / analysis_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load dataset (assumes it's stored in /app/datasets)
            dataset_path = Path(f"/app/datasets/{dataset_id}.parquet")
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            df = pd.read_parquet(dataset_path)
            logger.info(f"Loaded dataset: {df.shape}")
            
            # Step 1: Extract condition column
            condition_cols = [col for col in df.columns if 'Factor' in col or 'Condition' in col]
            if not condition_cols:
                raise ValueError("No condition column found in dataset")
            
            condition_col_name = condition_cols[0]
            conditions = df[condition_col_name].copy()
            logger.info(f"Found condition column: {condition_col_name}")
            logger.info(f"Condition values: {conditions.value_counts().to_dict()}")
            
            # Step 2: Drop condition column
            df = df.drop(columns=condition_cols)
            logger.info(f"After dropping condition: {df.shape}")
            
            # Step 3: Transpose to genes × samples (DESeq2 expects genes as rows)
            df = df.T
            logger.info(f"After transpose: {df.shape} (genes × samples)")
            
            # Step 4: Save count matrix (genes × samples, NO condition column)
            count_matrix_path = output_dir / "count_matrix.csv"
            df.to_csv(count_matrix_path)
            logger.info(f"Saved count matrix to {count_matrix_path}")
            
            # Step 5: Save condition metadata with sample names
            condition_df = pd.DataFrame({
                'sample': conditions.index,
                'condition': conditions.values
            })
            condition_path = output_dir / "condition_metadata.csv"
            condition_df.to_csv(condition_path, index=False)
            logger.info(f"Saved condition metadata to {condition_path}")
            logger.info(f"  Samples: {list(condition_df['sample'])}")
            logger.info(f"  Conditions: {list(condition_df['condition'])}")
            
            # Step 6: Call R function with condition file path
            r_code = f'''
result <- run_deseq2(
    "{str(count_matrix_path)}",
    "{str(condition_path)}",
    "{control_group}",
    "{treatment_group}",
    "{str(output_dir)}",
    {float(padj_threshold)},
    {float(log2fc_threshold)}
)
# Return just numeric values as a vector
c(result$num_genes, result$num_significant, result$num_upregulated, result$num_downregulated)
            '''
            
            # Execute R code and get numeric vector
            result_vec = ro.r(r_code)
            
            # Extract values from the vector
            num_genes = int(result_vec[0])
            num_significant = int(result_vec[1])
            num_upregulated = int(result_vec[2])
            num_downregulated = int(result_vec[3])
            
            logger.info(f"DESeq2 results: {num_genes} genes, {num_significant} significant")
            
            # Load differential genes from the CSV file R created
            results_file = output_dir / "deseq2_all_results.csv"
            if not results_file.exists():
                logger.error(f"Results file not found: {results_file}")
                differential_genes = []
            else:
                results_df = pd.read_csv(results_file, index_col=0)
                
                # Filter significant genes
                sig_genes = results_df[
                    (results_df['padj'] < padj_threshold) & 
                    (abs(results_df['log2FoldChange']) > log2fc_threshold)
                ].sort_values('padj')
                
                differential_genes = []
                for idx, (gene_id, row) in enumerate(sig_genes.iterrows(), 1):
                    differential_genes.append({
                        "gene_id": gene_id,
                        "log2_fold_change": float(row['log2FoldChange']),
                        "pvalue": float(row['pvalue']) if not pd.isna(row['pvalue']) else 1.0,
                        "padj": float(row['padj']) if not pd.isna(row['padj']) else 1.0,
                        "base_mean": float(row['baseMean']),
                        "rank": idx
                    })
            
            results = {
                "analysis_id": analysis_id,
                "num_genes": num_genes,
                "num_significant": num_significant,
                "num_upregulated": num_upregulated,
                "num_downregulated": num_downregulated,
                "differential_genes": differential_genes[:500],  # Top 500
                "volcano_plot_path": str(output_dir / "volcano_plot.png"),
                "ma_plot_path": str(output_dir / "ma_plot.png")
            }
            
            logger.info(f"✓ DESeq2 complete: {num_significant} significant genes")
            
            return results
            
        except Exception as e:
            logger.error(f"DESeq2 analysis failed: {e}", exc_info=True)
            raise
