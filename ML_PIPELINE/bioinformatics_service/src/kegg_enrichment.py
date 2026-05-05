# bioinformatics_service/src/kegg_enrichment.py
# bioinformatics_service/src/kegg_enrichment.py

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List

from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

logger = logging.getLogger(__name__)

class KEGGEnrichmentAnalyzer:
    """Wrapper for KEGG pathway enrichment analysis"""
    
    def __init__(self, output_base_path: str = "/app/results"):
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        self.r_script_path = Path("/app/r_scripts/kegg_enrichment.R")
        
        # Source R script ONCE at initialization
        r.source(str(self.r_script_path))
        logger.info(f"R script sourced: {self.r_script_path}")

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
            
            # Get the function from R global environment
            run_kegg_func = ro.globalenv['run_kegg_enrichment']
            
            # Convert parameters to R objects explicitly
            from rpy2.robjects import StrVector, FloatVector
            
            r_gene_list_path = str(gene_list_path)
            r_organism = str(organism)
            r_output_dir = str(output_dir)
            r_pvalue = float(pvalue_cutoff)
            r_qvalue = float(qvalue_cutoff)
            
            # Call R function with localconverter
            with localconverter(pandas2ri.converter):
                '''num_pathways = run_kegg_func(
                    r_gene_list_path,
                    r_organism,
                    r_output_dir,
                    r_pvalue,
                    r_qvalue
                )[0]'''

            # Use r() to call the function with string interpolation
            r_code = f'''
            run_kegg_enrichment(
                "{str(gene_list_path)}",
                "{str(organism)}",
                "{str(output_dir)}",
                {float(pvalue_cutoff)},
                {float(qvalue_cutoff)}
            )
            '''
            num_pathways = r(r_code)[0]            

            # Load gene conversion results
            conversion_file = output_dir / "gene_id_conversion.csv"
            if conversion_file.exists():
                conversion_df = pd.read_csv(conversion_file)
                logger.info(f"Gene ID conversion: {len(conversion_df)} genes converted")
                logger.info(f"  Conversion rate: {len(conversion_df) / len(gene_list) * 100:.1f}%")
            
            # Load results (outside converter context)
            pathways = []
            if num_pathways > 0:
                results_df = pd.read_csv(output_dir / "kegg_enrichment_results.csv")
                
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
            
            results = {
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
