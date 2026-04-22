# test_bioinformatics.py
import grpc
import sys
from pathlib import Path
import pandas as pd

# Add path
bio_service_path = Path(__file__).parent / "bioinformatics_service"
sys.path.insert(0, str(bio_service_path))

from generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc

def test_deseq2():
    """Test DESeq2 analysis"""
    
    # Connect to service
    channel = grpc.insecure_channel('localhost:50054')
    stub = bioinformatics_service_pb2_grpc.BioinformaticsServiceStub(channel)
    
    # First, you need a dataset with count data and conditions
    # For this test, assume you have a dataset already uploaded/downloaded
    #DATASET_ID = "d097a8c6-beed-40cf-84d0-ae24e2e3ee87" # osd-511
    DATASET_ID = "90519408-21d0-46b8-9081-0fe93ae10e30" # osd-48

    
    padj_threshold=0.9
    log2fc_threshold=5
    # Run DESeq2
    request = bioinformatics_service_pb2.DESeq2Request(
        dataset_id=DATASET_ID,
        condition_column="Factor Value[Spaceflight]",  # Your condition column
        control_group="Ground Control",
        treatment_group="Space Flight",
        padj_threshold=padj_threshold,
        log2fc_threshold=log2fc_threshold,
    )
    
    print("Running DESeq2 analysis...")
    response = stub.RunDESeq2(request)
    
    if response.success:
        print(f"\n✓ DESeq2 Analysis Complete!")
        print(f"  Analysis ID: {response.analysis_id}")
        print(f"\nResults:")
        print(f"  Total genes: {response.results.num_genes}")
        print(f"  Significant genes: {response.results.num_significant}")
        print(f"  Upregulated: {response.results.num_upregulated}")
        print(f"  Downregulated: {response.results.num_downregulated}")
        
        print(f"\nTop 10 Differentially Expressed Genes:")
        for gene in response.results.differential_genes[:100]:
            #if gene.log2_fold_change >= log2fc_threshold and gene.padj <= padj_threshold:
            if gene.padj <= padj_threshold:
                print(f"  {gene.gene_id}: log2FC={gene.log2_fold_change:.2f}, padj={gene.padj:.2e}")
        
        print(f"\nPlots:")
        print(f"  Volcano plot: {response.results.volcano_plot_path}")
        print(f"  MA plot: {response.results.ma_plot_path}")
        
        # Now run KEGG enrichment on these results
        print("\n" + "="*60)
        print("Running KEGG Enrichment...")
        kegg_request = bioinformatics_service_pb2.KEGGRequest(
            analysis_id=response.analysis_id,
            organism="mmu",  # Mouse (use "hsa" for human)
            pvalue_cutoff=0.9,
            qvalue_cutoff=0.9
        )
        
        kegg_response = stub.RunKEGGEnrichment(kegg_request)
        
        if kegg_response.success:
            print(f"\n✓ KEGG Enrichment Complete!")
            print(f"  Enriched pathways: {kegg_response.results.num_pathways}")
            
            if kegg_response.results.num_pathways > 0:
                print(f"\nTop 10 Enriched KEGG Pathways:")
                for pathway in kegg_response.results.pathways[:10]:
                    print(f"  {pathway.pathway_id}: {pathway.description}")
                    print(f"    P-value: {pathway.pvalue:.2e}, Gene count: {pathway.gene_count}")
                
                print(f"\nPlots:")
                print(f"  Dotplot: {kegg_response.results.dotplot_path}")
                print(f"  Barplot: {kegg_response.results.barplot_path}")
            else:
                print("  No significantly enriched pathways found")
        else:
            print(f"✗ KEGG enrichment failed: {kegg_response.error_message}")
    else:
        print(f"✗ DESeq2 analysis failed: {response.error_message}")

if __name__ == "__main__":
    test_deseq2()
