import grpc
import logging
from typing import Dict, List
from src.generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc

logger = logging.getLogger(__name__)


class BioinformaticsClient:
    """Client for bioinformatics service (DESeq2, KEGG)"""
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel = grpc.insecure_channel(service_url)
        self.stub = bioinformatics_service_pb2_grpc.BioinformaticsServiceStub(self.channel)
        logger.info(f"Bioinformatics client initialized: {service_url}")
    
    def run_deseq2(
        self,
        dataset_id: str,
        condition_column: str,
        control_group: str,
        treatment_group: str,
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 0.0
    ) -> Dict:
        """Run DESeq2 differential expression analysis"""
        try:
            request = bioinformatics_service_pb2.DESeq2Request(
                dataset_id=dataset_id,
                condition_column=condition_column,
                control_group=control_group,
                treatment_group=treatment_group,
                padj_threshold=padj_threshold,
                log2fc_threshold=log2fc_threshold
            )
            
            response = self.stub.RunDESeq2(request)
            
            if response.success and response.results:
                results = response.results
                
                # Extract differential genes
                differential_genes = []
                for gene in results.differential_genes:
                    differential_genes.append({
                        "gene_id": gene.gene_id,
                        "log2_fold_change": float(gene.log2_fold_change),
                        "pvalue": float(gene.pvalue),
                        "padj": float(gene.padj),
                        "base_mean": float(gene.base_mean),
                        "rank": int(gene.rank)
                    })
                
                return {
                    "success": True,
                    "analysis_id": response.analysis_id,
                    "num_genes": results.num_genes,
                    "num_significant": results.num_significant,
                    "num_upregulated": results.num_upregulated,
                    "num_downregulated": results.num_downregulated,
                    "differential_genes": differential_genes,
                    "volcano_plot_path": results.volcano_plot_path,
                    "ma_plot_path": results.ma_plot_path
                }
            else:
                return {
                    "success": False,
                    "error": response.error_message
                }
        
        except Exception as e:
            logger.error(f"Error running DESeq2: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_kegg_enrichment(
        self,
        analysis_id: str,
        gene_list: List[str],
        organism: str = "mmu",
        pvalue_cutoff: float = 0.05,
        qvalue_cutoff: float = 0.05
    ) -> Dict:
        """Run KEGG pathway enrichment analysis"""
        try:
            request = bioinformatics_service_pb2.KEGGRequest(
                analysis_id=analysis_id,
                gene_list=gene_list,
                organism=organism,
                pvalue_cutoff=pvalue_cutoff,
                qvalue_cutoff=qvalue_cutoff
            )
            
            response = self.stub.RunKEGGEnrichment(request)
            
            if response.success and response.results:
                results = response.results
                
                # Extract pathways
                pathways = []
                for pathway in results.pathways:
                    pathways.append({
                        "pathway_id": pathway.pathway_id,
                        "description": pathway.description,
                        "pvalue": float(pathway.pvalue),
                        "padj": float(pathway.padj),
                        "gene_count": pathway.gene_count,
                        "gene_ratio": pathway.gene_ratio,
                        "bg_ratio": pathway.bg_ratio,
                        "genes": list(pathway.genes),
                        "rank": pathway.rank
                    })
                
                return {
                    "success": True,
                    "num_pathways": results.num_pathways,
                    "pathways": pathways,
                    "dotplot_path": results.dotplot_path,
                    "barplot_path": results.barplot_path
                }
            else:
                return {
                    "success": False,
                    "error": response.error_message
                }
        
        except Exception as e:
            logger.error(f"Error running KEGG enrichment: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def close(self):
        self.channel.close()
