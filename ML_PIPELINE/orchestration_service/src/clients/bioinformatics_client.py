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
        # Enrichment requests/responses carry thousands of genes and terms: allow larger messages
        self.enrich_channel = grpc.insecure_channel(service_url, options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
        ])
        self.enrich_stub = bioinformatics_service_pb2_grpc.BioinformaticsServiceStub(self.enrich_channel)
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
    
    def run_go_enrichment(
        self,
        gene_lists: Dict[str, List[str]],
        rankings: Dict[str, Dict],          # name -> {"genes": [...], "scores": [...], "score_type": "std"|"pos"}
        universe: List[str],
        organism: str = "mmu",
        ontologies: List[str] = None,
        p_adjust_method: str = "BH",
        padj_cutoff: float = 0.05,
        qvalue_cutoff: float = 0.2,
        min_gs_size: int = 10,
        max_gs_size: int = 500,
        simplify: bool = True,
        simplify_cutoff: float = 0.7,
        analysis_id: str = "",
        timeout: float = 1800,
    ) -> Dict:
        """GO enrichment: ORA on each gene list, GSEA on each ranking"""
        try:
            request = bioinformatics_service_pb2.EnrichmentRequest(
                analysis_id=analysis_id or "",
                organism=organism,
                gene_lists=[bioinformatics_service_pb2.GeneList(name=n, genes=list(g)) for n, g in gene_lists.items()],
                rankings=[bioinformatics_service_pb2.GeneRanking(
                    name=n, genes=list(r["genes"]), scores=[float(s) for s in r["scores"]],
                    score_type=r.get("score_type", "std")) for n, r in rankings.items()],
                universe=list(universe),
                ontologies=list(ontologies or ["BP"]),
                p_adjust_method=p_adjust_method,
                padj_cutoff=padj_cutoff,
                qvalue_cutoff=qvalue_cutoff,
                min_gs_size=min_gs_size,
                max_gs_size=max_gs_size,
                simplify=simplify,
                simplify_cutoff=simplify_cutoff,
            )
            response = self.enrich_stub.RunGOEnrichment(request, timeout=timeout)
            if not response.success:
                return {"success": False, "error": response.error_message}
            return {
                "success": True,
                "output_dir": response.output_dir,
                "terms": [{
                    "analysis": t.analysis, "method": t.method, "ontology": t.ontology,
                    "term_id": t.term_id, "description": t.description,
                    "set_size": t.set_size, "count": t.count,
                    "gene_ratio": t.gene_ratio, "bg_ratio": t.bg_ratio,
                    "enrichment_score": t.enrichment_score, "nes": t.nes,
                    "pvalue": t.pvalue, "padj": t.padj, "qvalue": t.qvalue,
                    "genes": list(t.genes),
                } for t in response.terms],
                "summaries": [{
                    "analysis": s.analysis, "method": s.method, "ontology": s.ontology,
                    "n_input": s.n_input, "n_mapped": s.n_mapped, "n_terms": s.n_terms, "note": s.note,
                } for s in response.summaries],
            }
        except Exception as e:
            logger.error(f"Error running GO enrichment: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def close(self):
        self.channel.close()
        self.enrich_channel.close()
