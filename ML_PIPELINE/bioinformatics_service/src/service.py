# bioinformatics_service/src/service.py

import grpc
import sys
from pathlib import Path
import logging
import pandas as pd
from typing import Dict

# Add data_service path to import data_client
data_service_path = Path(__file__).parent.parent.parent / "data_service"
sys.path.insert(0, str(data_service_path))

from src.data_client import DataServiceClient

from generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc
from src.deseq2_analysis import DESeq2Analyzer
from src.kegg_enrichment import KEGGEnrichmentAnalyzer

logger = logging.getLogger(__name__)

class BioinformaticsServiceImpl(bioinformatics_service_pb2_grpc.BioinformaticsServiceServicer):
    """gRPC service for bioinformatics analysis"""
    
    def __init__(self, data_service_url: str = "data_service:50051"):
        self.data_client = DataServiceClient(data_service_url)
        self.deseq2_analyzer = DESeq2Analyzer()
        self.kegg_analyzer = KEGGEnrichmentAnalyzer()
        
        # Cache analysis results
        self.analysis_cache: Dict[str, Dict] = {}
        
        logger.info("BioinformaticsService initialized")
    
    def RunDESeq2(self, request, context):
        """Run DESeq2 differential expression analysis"""
        try:
            dataset_id = request.dataset_id
            condition_column = request.condition_column
            control_group = request.control_group
            treatment_group = request.treatment_group
            padj_threshold = request.padj_threshold or 0.05
            log2fc_threshold = request.log2fc_threshold or 1.0
            covariates = list(request.covariates) if request.covariates else []
            
            logger.info(f"DESeq2 request: {treatment_group} vs {control_group}")
            logger.info(f"  Dataset: {dataset_id}")
            logger.info(f"  Condition column: {condition_column}")
            
            # Get dataset from data service
            df = self.data_client.get_dataset(dataset_id)
            if df is None:
                return bioinformatics_service_pb2.DESeq2Response(
                    success=False,
                    error_message=f"Dataset {dataset_id} not found"
                )
            
            logger.info(f"Dataset loaded: {df.shape}")
            
            # Separate count data and metadata
            # Assume: condition_column is in the dataset
            # Count columns are numeric, metadata columns are categorical
            
            if condition_column not in df.columns:
                return bioinformatics_service_pb2.DESeq2Response(
                    success=False,
                    error_message=f"Condition column '{condition_column}' not found in dataset"
                )
            
            # Extract metadata
            metadata_columns = [condition_column] + covariates
            metadata = df[metadata_columns].copy()
            
            # Extract count data (all numeric columns except metadata)
            count_columns = [col for col in df.columns if col not in metadata_columns]
            count_data = df[count_columns].copy()
            
            # Transpose if needed (genes should be rows, samples should be columns)
            # DESeq2 expects: rows = genes, columns = samples
            if count_data.shape[0] < count_data.shape[1]:
                logger.info("Transposing count matrix (detected samples as rows)")
                count_data = count_data.T
            
            logger.info(f"Count data: {count_data.shape} (genes x samples)")
            logger.info(f"Metadata: {metadata.shape}")
            
            # Run DESeq2
            results = self.deseq2_analyzer.run_analysis(
                count_data=count_data,
                metadata=metadata,
                condition_column=condition_column,
                control_group=control_group,
                treatment_group=treatment_group,
                padj_threshold=padj_threshold,
                log2fc_threshold=log2fc_threshold,
                covariates=covariates
            )
            
            # Cache results for later KEGG analysis
            analysis_id = results["analysis_id"]
            self.analysis_cache[analysis_id] = results
            
            # Convert to protobuf
            de_genes = []
            for gene in results["differential_genes"]:
                de_genes.append(
                    bioinformatics_service_pb2.DEGene(
                        gene_id=gene["gene_id"],
                        base_mean=gene["base_mean"],
                        log2_fold_change=gene["log2_fold_change"],
                        lfcse=gene["lfcse"],
                        stat=gene["stat"],
                        pvalue=gene["pvalue"],
                        padj=gene["padj"],
                        rank=gene["rank"]
                    )
                )
            
            deseq2_results = bioinformatics_service_pb2.DESeq2Results(
                num_genes=results["num_genes"],
                num_upregulated=results["num_upregulated"],
                num_downregulated=results["num_downregulated"],
                num_significant=results["num_significant"],
                differential_genes=de_genes,
                volcano_plot_path=results["volcano_plot_path"],
                ma_plot_path=results["ma_plot_path"],
                metadata={
                    "results_path": results["results_path"],
                    "significant_genes_path": results["significant_genes_path"]
                }
            )
            
            return bioinformatics_service_pb2.DESeq2Response(
                success=True,
                analysis_id=analysis_id,
                results=deseq2_results
            )
            
        except Exception as e:
            logger.error(f"DESeq2 error: {e}", exc_info=True)
            return bioinformatics_service_pb2.DESeq2Response(
                success=False,
                error_message=str(e)
            )
    
    def RunKEGGEnrichment(self, request, context):
        """Run KEGG pathway enrichment analysis"""
        try:
            logger.info(f" request in RunKEGGEnrichment: {request}")
            analysis_id = request.analysis_id
            gene_list = list(request.gene_list) if request.gene_list else []
            organism = request.organism or "mmu"
            pvalue_cutoff = request.pvalue_cutoff or 0.05
            qvalue_cutoff = request.qvalue_cutoff or 0.2
            
            logger.info(f"KEGG enrichment request")
            logger.info(f"  Analysis ID: {analysis_id}")
            logger.info(f"  Organism: {organism}")
            
            # Get gene list from DESeq2 results or use provided list
            if analysis_id and analysis_id in self.analysis_cache:
                # Extract significant genes from DESeq2 results
                sig_genes_path = self.analysis_cache[analysis_id]["significant_genes_path"]
                sig_df = pd.read_csv(sig_genes_path, index_col=0)
                gene_list = sig_df.index.tolist()
                logger.info(f"Using {len(gene_list)} significant genes from DESeq2")
            elif not gene_list:
                return bioinformatics_service_pb2.KEGGResponse(
                    success=False,
                    error_message="No gene list provided and analysis_id not found"
                )
            
            if len(gene_list) == 0:
                return bioinformatics_service_pb2.KEGGResponse(
                    success=False,
                    error_message="Gene list is empty"
                )
            
            # Run KEGG enrichment
            results = self.kegg_analyzer.run_enrichment(
                gene_list=gene_list,
                organism=organism,
                pvalue_cutoff=pvalue_cutoff,
                qvalue_cutoff=qvalue_cutoff
            )
            
            # Convert to protobuf
            pathways = []
            for pathway in results["pathways"]:
                pathways.append(
                    bioinformatics_service_pb2.KEGGPathway(
                        pathway_id=pathway["pathway_id"],
                        description=pathway["description"],
                        pvalue=pathway["pvalue"],
                        padj=pathway["padj"],
                        gene_count=pathway["gene_count"],
                        gene_ratio=float(pathway["gene_ratio"].split("/")[0]) / float(pathway["gene_ratio"].split("/")[1]) if "/" in str(pathway["gene_ratio"]) else 0.0,
                        bg_ratio=int(pathway["bg_ratio"].split("/")[0]) if "/" in str(pathway["bg_ratio"]) else 0,
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
    
    def RunGSEA(self, request, context):
        """Run Gene Set Enrichment Analysis (GSEA)"""
        # TODO: Implement GSEA
        return bioinformatics_service_pb2.GSEAResponse(
            success=False,
            error_message="GSEA not yet implemented"
        )
