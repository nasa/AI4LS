# src/service.py

import grpc
import logging
import os
from pathlib import Path

from src import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc
from src.kegg_enrichment import KEGGEnrichment
from src.go_enrichment import GOEnrichment
from src.deseq2_analysis import DESeq2Analysis
from src.data_client import DataServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import threading
R_LOCK = threading.Lock()

class BioinformaticsServiceImpl(bioinformatics_service_pb2_grpc.BioinformaticsServiceServicer):
    """gRPC service for bioinformatics analysis"""
    
    def __init__(self, results_path: str = "/app/results"):
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Initialize data service client
        self.data_client = DataServiceClient(service_url="data_service:50051")
        
        self.kegg_analyzer = KEGGEnrichment(results_path)
        self.go_analyzer = GOEnrichment(results_path)
        self.deseq2_analyzer = DESeq2Analysis(results_path, self.data_client)

    
        
        logger.info("BioinformaticsService initialized")
        logger.info(f"Results path: {self.results_path}")
    
    def RunDESeq2(self, request, context):
        """Run DESeq2 differential expression analysis"""
        try:
            logger.info(f"Running DESeq2 for dataset {request.dataset_id}")
            with R_LOCK: 
                results = self.deseq2_analyzer.run_analysis(
                    dataset_id=request.dataset_id,
                    condition_column=request.condition_column,
                    control_group=request.control_group,
                    treatment_group=request.treatment_group,
                    padj_threshold=request.padj_threshold or 0.05,
                    log2fc_threshold=request.log2fc_threshold or 2.0
                )
            
                # Build response
                differential_genes = []
                for gene in results["differential_genes"]:
                    differential_genes.append(
                        bioinformatics_service_pb2.DifferentialGene(
                            gene_id=gene["gene_id"],
                            log2_fold_change=gene["log2_fold_change"],
                            pvalue=gene["pvalue"],
                            padj=gene["padj"],
                            base_mean=gene["base_mean"],
                            rank=gene["rank"]
                        )
                    )
            
                deseq2_results = bioinformatics_service_pb2.DESeq2Results(
                    num_genes=results["num_genes"],
                    num_significant=results["num_significant"],
                    num_upregulated=results["num_upregulated"],
                    num_downregulated=results["num_downregulated"],
                    differential_genes=differential_genes,
                    volcano_plot_path=results["volcano_plot_path"],
                    ma_plot_path=results["ma_plot_path"]
                )
                
                return bioinformatics_service_pb2.DESeq2Response(
                    success=True,
                    analysis_id=results["analysis_id"],
                    results=deseq2_results
                )
            
        except Exception as e:
            logger.error(f"DESeq2 analysis error: {e}", exc_info=True)
            return bioinformatics_service_pb2.DESeq2Response(
                success=False,
                error_message=str(e)
            )
    
    def RunKEGGEnrichment(self, request, context):
        """Run KEGG pathway enrichment"""
        try:
            with R_LOCK:
                logger.info(f"Running KEGG enrichment for analysis {request.analysis_id}")
                
                results = self.kegg_analyzer.run_enrichment(
                    gene_list=list(request.gene_list),
                    organism=request.organism or "mmu",
                    pvalue_cutoff=request.pvalue_cutoff or 0.05,
                    qvalue_cutoff=request.qvalue_cutoff or 0.2,
                    analysis_id=request.analysis_id
                )
            
                # Build response
                pathways = []
                for pathway in results["pathways"]:
                    pathways.append(
                        bioinformatics_service_pb2.KEGGPathway(
                            pathway_id=pathway["pathway_id"],
                            description=pathway["description"],
                            pvalue=pathway["pvalue"],
                            padj=pathway["padj"],
                            gene_count=pathway["gene_count"],
                            gene_ratio=pathway["gene_ratio"],
                            bg_ratio=pathway["bg_ratio"],
                            genes=pathway["genes"],
                            rank=pathway["rank"]
                        )
                    )
            
                kegg_results = bioinformatics_service_pb2.KEGGResults(
                    num_pathways=results["num_pathways"],
                    pathways=pathways,
                    dotplot_path=results["dotplot_path"],
                    barplot_path=results["barplot_path"]
                )
            
                return bioinformatics_service_pb2.KEGGResponse(
                    success=True,
                    results=kegg_results
                )
            
        except Exception as e:
            logger.error(f"KEGG enrichment error: {e}", exc_info=True)
            return bioinformatics_service_pb2.KEGGResponse(
                success=False,
                error_message=str(e)
            )

    def RunGOEnrichment(self, request, context):
        """GO enrichment: ORA on each gene list, GSEA on each ranking"""
        try:
            gene_lists = {gl.name: list(gl.genes) for gl in request.gene_lists}
            rankings = {r.name: {"genes": list(r.genes), "scores": list(r.scores),
                                 "score_type": r.score_type or "std"} for r in request.rankings}
            if not gene_lists and not rankings:
                return bioinformatics_service_pb2.EnrichmentResponse(
                    success=False, error_message="No gene lists or rankings to analyse")
            for name, r in rankings.items():
                if len(r["genes"]) != len(r["scores"]):
                    return bioinformatics_service_pb2.EnrichmentResponse(
                        success=False, error_message=f"Ranking '{name}': {len(r['genes'])} genes but {len(r['scores'])} scores")

            with R_LOCK:   # embedded R is not thread-safe
                results = self.go_analyzer.run(
                    gene_lists=gene_lists,
                    rankings=rankings,
                    universe=list(request.universe),
                    organism=request.organism or "mmu",
                    ontologies=list(request.ontologies) or ["BP"],
                    p_adjust_method=request.p_adjust_method or "BH",
                    padj_cutoff=request.padj_cutoff or 0.05,
                    qvalue_cutoff=request.qvalue_cutoff or 0.2,
                    min_gs_size=request.min_gs_size or 10,
                    max_gs_size=request.max_gs_size or 500,
                    simplify=request.simplify,
                    simplify_cutoff=request.simplify_cutoff or 0.7,
                    analysis_id=request.analysis_id or None,
                )

            return bioinformatics_service_pb2.EnrichmentResponse(
                success=True,
                terms=[bioinformatics_service_pb2.EnrichedTerm(**t) for t in results["terms"]],
                summaries=[bioinformatics_service_pb2.EnrichmentSummary(**s) for s in results["summaries"]],
                output_dir=results["output_dir"],
            )
        except Exception as e:
            logger.error(f"GO enrichment error: {e}", exc_info=True)
            return bioinformatics_service_pb2.EnrichmentResponse(success=False, error_message=str(e))
