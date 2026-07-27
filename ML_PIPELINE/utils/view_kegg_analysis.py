# view_kegg_analyses.py

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def main():
    results_dir = Path("results")
    
    # Find all KEGG analysis directories
    kegg_dirs = list(results_dir.glob("kegg_*"))
    
    if not kegg_dirs:
        print("No KEGG analyses found")
        return
    
    if len(sys.argv) > 1:
        # Show detailed view for specific analysis
        analysis_pattern = sys.argv[1]
        matching_dirs = [d for d in kegg_dirs if analysis_pattern in d.name]
        
        if not matching_dirs:
            print(f"No KEGG analysis found matching: {analysis_pattern}")
            return
        
        analysis_dir = matching_dirs[0]
        
        print("\n" + "=" * 100)
        print(f"KEGG ANALYSIS: {analysis_dir.name}")
        print("=" * 100)
        
        # Load gene list
        gene_list_file = analysis_dir / "gene_list.csv"
        if gene_list_file.exists():
            gene_list = pd.read_csv(gene_list_file)
            print(f"\nInput Genes: {len(gene_list)}")
            print(f"First 10 genes:")
            for i, gene in enumerate(gene_list['gene_id'].head(10), 1):
                print(f"  {i:3d}. {gene}")
        
        # Load gene conversion
        conversion_file = analysis_dir / "gene_id_conversion.csv"
        if conversion_file.exists():
            conversion = pd.read_csv(conversion_file)
            print(f"\nGene ID Conversion:")
            print(f"  Input genes: {len(gene_list) if gene_list_file.exists() else 'N/A'}")
            print(f"  Converted genes: {len(conversion)}")
            if len(gene_list) > 0:
                conversion_rate = len(conversion) / len(gene_list) * 100
                print(f"  Conversion rate: {conversion_rate:.1f}%")
            
            print(f"\nConversion Examples:")
            print(conversion.head(10).to_string(index=False))
        
        # Load KEGG results
        kegg_results_file = analysis_dir / "kegg_enrichment_results.csv"
        if kegg_results_file.exists():
            kegg_results = pd.read_csv(kegg_results_file)
            
            print(f"\nEnriched Pathways: {len(kegg_results)}")
            
            if len(kegg_results) > 0:
                print(f"\nTop 20 Enriched KEGG Pathways:")
                print(f"{'Rank':<6} {'Pathway ID':<12} {'Description':<50} {'P-value':<12} {'Genes':<8}")
                print("-" * 100)
                
                for i, row in kegg_results.head(20).iterrows():
                    rank = i + 1
                    pathway_id = row['ID']
                    description = row['Description'][:47] + "..." if len(row['Description']) > 50 else row['Description']
                    pvalue = f"{row['pvalue']:.2e}"
                    gene_count = row['Count']
                    
                    print(f"{rank:<6} {pathway_id:<12} {description:<50} {pvalue:<12} {gene_count:<8}")
                
                print(f"\nPathway Statistics:")
                print(f"  Most significant p-value: {kegg_results['pvalue'].min():.2e}")
                print(f"  Median gene count: {kegg_results['Count'].median():.0f}")
                print(f"  Max genes in pathway: {kegg_results['Count'].max():.0f}")
                
                # Show most significant pathway details
                print(f"\nMost Significant Pathway:")
                top_pathway = kegg_results.iloc[0]
                print(f"  ID: {top_pathway['ID']}")
                print(f"  Description: {top_pathway['Description']}")
                print(f"  P-value: {top_pathway['pvalue']:.2e}")
                print(f"  Adjusted P-value: {top_pathway['p.adjust']:.2e}")
                print(f"  Gene Ratio: {top_pathway['GeneRatio']}")
                print(f"  BG Ratio: {top_pathway['BgRatio']}")
                print(f"  Genes: {top_pathway['geneID'][:100]}...")
            else:
                print("\n  No enriched pathways found")
        else:
            print("\nNo KEGG enrichment results found")
        
        # Show available plots
        print(f"\nAvailable Visualizations:")
        dotplot = analysis_dir / "kegg_dotplot.png"
        barplot = analysis_dir / "kegg_barplot.png"
        
        if dotplot.exists():
            print(f"  ✓ Dotplot: {dotplot}")
        else:
            print(f"  ✗ Dotplot not found")
        
        if barplot.exists():
            print(f"  ✓ Barplot: {barplot}")
        else:
            print(f"  ✗ Barplot not found")
        
        print(f"\nAnalysis Directory: {analysis_dir}")
    
    else:
        # List all KEGG analyses
        print("\n" + "=" * 100)
        print("KEGG PATHWAY ENRICHMENT ANALYSES")
        print("=" * 100)
        print(f"{'Analysis ID':<45} {'Organism':<10} {'Genes':<8} {'Pathways':<10} {'Modified'}")
        print("-" * 100)
        
        for kegg_dir in sorted(kegg_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
            analysis_id = kegg_dir.name
            
            # Extract organism
            if "_mmu_" in analysis_id:
                organism = "mmu"
            elif "_hsa_" in analysis_id:
                organism = "hsa"
            else:
                organism = "unknown"
            
            # Count genes
            gene_list_file = kegg_dir / "gene_list.csv"
            num_genes = 0
            if gene_list_file.exists():
                gene_list = pd.read_csv(gene_list_file)
                num_genes = len(gene_list)
            
            # Count pathways
            kegg_results_file = kegg_dir / "kegg_enrichment_results.csv"
            num_pathways = 0
            if kegg_results_file.exists():
                kegg_results = pd.read_csv(kegg_results_file)
                num_pathways = len(kegg_results)
            
            # Modified time
            modified = datetime.fromtimestamp(kegg_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            
            print(f"{analysis_id:<45} {organism:<10} {num_genes:<8} {num_pathways:<10} {modified}")
        
        print("-" * 100)
        print(f"Total: {len(kegg_dirs)} analyses")
        print("\nTo view detailed results for an analysis, run:")
        print("  python view_kegg_analyses.py <analysis_id>")
        print("  python view_kegg_analyses.py <partial_match>")
        print("\nExample:")
        print("  python view_kegg_analyses.py kegg_mmu_model_abc123")
        print("  python view_kegg_analyses.py model_abc123")

if __name__ == "__main__":
    main()
