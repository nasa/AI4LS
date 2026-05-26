# src/kegg_enrichment.py

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KEGGEnrichment:
    """KEGG pathway enrichment using R clusterProfiler"""
    
    def __init__(self, output_base_path: str = "/app/results"):
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        # Source R script once during initialization
        r_script_path = Path("/app/r_scripts/kegg_enrichment.R")
        if not r_script_path.exists():
            raise FileNotFoundError(f"R script not found: {r_script_path}")
        
        ro.r(f'source("{r_script_path}")')
        logger.info(f"Sourced R script: {r_script_path}")
    
    def run_enrichment(
        self,
        gene_list: List[str],
        organism: str = "mmu",
        pvalue_cutoff: float = 0.05,
        qvalue_cutoff: float = 0.2,
        analysis_id: str = None
    ) -> Dict:
        """Run KEGG pathway enrichment analysis"""
        try:
            logger.info(f"Running KEGG enrichment for {len(gene_list)} genes")
            logger.info(f"  Organism: {organism}")
            logger.info(f"  Analysis ID: {analysis_id}")
            logger.info(f"  Gene IDs (first 5): {gene_list[:5]}")
            
            # Create output directory with analysis_id if provided
            if analysis_id:
                output_dir = self.output_base_path / f"kegg_{organism}_{analysis_id}"
            else:
                output_dir = self.output_base_path / f"kegg_{organism}"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            
            # Save gene list
            gene_list_path = output_dir / "gene_list.csv"
            pd.DataFrame({"gene_id": gene_list}).to_csv(gene_list_path, index=False)
            
            # Get the R function from global environment
            from rpy2.robjects import globalenv
            run_kegg_func = globalenv['run_kegg_enrichment']
            
            # Call the R function directly (avoids conversion issues)
            result = run_kegg_func(
                str(gene_list_path),
                str(organism),
                str(output_dir),
                float(pvalue_cutoff),
                float(qvalue_cutoff)
            )
            
            # Extract the number of pathways
            num_pathways = int(result[0]) if result is not None and len(result) > 0 else 0
            
            logger.info(f"R function returned: {num_pathways} pathways")
            
            # Load gene conversion results
            conversion_file = output_dir / "gene_id_conversion.csv"
            if conversion_file.exists():
                conversion_df = pd.read_csv(conversion_file)
                logger.info(f"Gene ID conversion: {len(conversion_df)} genes converted")
                logger.info(f"  Conversion rate: {len(conversion_df) / len(gene_list) * 100:.1f}%")
            
            # Load results
            pathways = []
            if num_pathways > 0:
                results_file = output_dir / "kegg_enrichment_results.csv"
                if results_file.exists():
                    results_df = pd.read_csv(results_file)
                    
                    for idx, row in results_df.iterrows():
                        pathways.append({
                            "pathway_id": row["ID"],
                            "description": row["Description"],
                            "pvalue": float(row["pvalue"]),
                            "padj": float(row["p.adjust"]),
                            "gene_count": int(row["Count"]),
                            "gene_ratio": row["GeneRatio"],
                            "bg_ratio": row["BgRatio"],
                            "genes": row["geneID"].split("/"),
                            "rank": idx + 1
                        })
                else:
                    logger.warning(f"Results file not found: {results_file}")
            
            results = {
                "success": True,
                "num_pathways": int(num_pathways),
                "pathways": pathways,
                "dotplot_path": str(output_dir / "kegg_dotplot.png") if num_pathways > 0 else "",
                "barplot_path": str(output_dir / "kegg_barplot.png") if num_pathways > 0 else "",
                "conversion_path": str(conversion_file) if conversion_file.exists() else ""
            }
            
            logger.info(f"✓ KEGG enrichment complete: {num_pathways} pathways enriched")
            
            return results
            
        except Exception as e:
            logger.error(f"KEGG enrichment failed: {e}", exc_info=True)
            raise
