# bioinformatics_service/src/deseq2_analysis.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import uuid
from typing import Dict, List

from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

logger = logging.getLogger(__name__)

class DESeq2Analyzer:
    """Wrapper for DESeq2 differential expression analysis"""
    
    def __init__(self, output_base_path: str = "/app/results"):
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        self.r_script_path = Path("/app/r_scripts/deseq2_wrapper.R")
        
        # Source R script ONCE at initialization
        r.source(str(self.r_script_path))
        logger.info(f"R script sourced: {self.r_script_path}")
        
    def run_analysis(
        self,
        count_data: pd.DataFrame,
        metadata: pd.DataFrame,
        condition_column: str,
        control_group: str,
        treatment_group: str,
        padj_threshold: float,
        log2fc_threshold: float,
        covariates: List[str] = None
    ) -> Dict:
        """Run DESeq2 differential expression analysis"""
        try:
            # Create analysis ID
            analysis_id = f"deseq2_{uuid.uuid4().hex[:12]}"
            output_dir = self.output_base_path / analysis_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Running DESeq2 analysis: {analysis_id}")
            logger.info(f"  Comparison: {treatment_group} vs {control_group}")
            logger.info(f"  Condition column: {condition_column}")
            logger.info(f"  Genes: {count_data.shape[0]}, Samples: {count_data.shape[1]}")

            logger.info(f"head of count_data.csv: {count_data.head()}")
            
            # Prepare input files
            count_matrix_path = output_dir / "count_matrix.csv"
            metadata_path = output_dir / "metadata.csv"

            logger.info(f"count_matrix head: {count_data.head()}") 
            logger.info(f"metadata head: {metadata.head()}") 
            
            count_data.to_csv(count_matrix_path)
            metadata.to_csv(metadata_path)
            
            # Call R function - use ro.globalenv to get the function
            logger.info("Calling R run_deseq2 function...")
            
            # Get the function from R global environment
            run_deseq2_func = ro.globalenv['run_deseq2']
            
            # Call it with localconverter
            #with localconverter(pandas2ri.converter):
            logger.info(f"calling run_deseq2_func with padj= {padj_threshold} and l2fc = {log2fc_threshold}")
            r_results = run_deseq2_func(
                str(count_matrix_path),
                str(metadata_path),
                condition_column,
                control_group,
                treatment_group,
                str(output_dir),
                float(padj_threshold),
                float(log2fc_threshold)
            )
                
            logger.info(f"R function returned: {type(r_results)}")
                
            # Extract results
            num_genes = int(r_results.rx2("num_genes")[0])
            num_upregulated = int(r_results.rx2("num_upregulated")[0])
            num_downregulated = int(r_results.rx2("num_downregulated")[0])
            num_significant = int(r_results.rx2("num_significant")[0])
        
            # load results files (outside converter context)
            all_results = pd.read_csv(output_dir / "deseq2_all_results.csv", index_col=0)
            
            # Check if significant genes file exists
            sig_genes_path = output_dir / "deseq2_significant_genes.csv"
            if sig_genes_path.exists():
                sig_results = pd.read_csv(sig_genes_path, index_col=0)
            
            # Parse top DE genes
            top_genes = []
            for idx, (gene_id, row) in enumerate(all_results.head(100).iterrows()):
                top_genes.append({
                    "gene_id": str(gene_id),
                    "base_mean": float(row["baseMean"]) if not pd.isna(row["baseMean"]) else 0.0,
                    "log2_fold_change": float(row["log2FoldChange"]) if not pd.isna(row["log2FoldChange"]) else 0.0,
                    "lfcse": float(row["lfcSE"]) if not pd.isna(row["lfcSE"]) else 0.0,
                    "stat": float(row["stat"]) if not pd.isna(row["stat"]) else 0.0,
                    "pvalue": float(row["pvalue"]) if not pd.isna(row["pvalue"]) else 1.0,
                    "padj": float(row["padj"]) if not pd.isna(row["padj"]) else 1.0,
                    "rank": idx + 1
                })
            
            results = {
                "analysis_id": analysis_id,
                "num_genes": num_genes,
                "num_upregulated": num_upregulated,
                "num_downregulated": num_downregulated,
                "num_significant": num_significant,
                "differential_genes": top_genes,
                "volcano_plot_path": str(output_dir / "volcano_plot.png"),
                "ma_plot_path": str(output_dir / "ma_plot.png"),
                "results_path": str(output_dir / "deseq2_all_results.csv"),
                "significant_genes_path": str(sig_genes_path) if sig_genes_path.exists() else ""
            }
            
            logger.info(f"✓ DESeq2 analysis complete")
            logger.info(f"  Total genes: {results['num_genes']}")
            logger.info(f"  Significant: {results['num_significant']}")
            logger.info(f"  Upregulated: {results['num_upregulated']}")
            logger.info(f"  Downregulated: {results['num_downregulated']}")
            
            return results
            
        except Exception as e:
            logger.error(f"DESeq2 analysis failed: {e}", exc_info=True)
            raise
