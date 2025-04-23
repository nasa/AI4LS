from mcp.server.fastmcp import FastMCP, Image
import httpx
import matplotlib.pyplot as plt
from io import StringIO
import os
import pandas as pd

mcp = FastMCP("OSDR_VIZ_SERVER")

# Set output directory, defaulting to `../../data/mcp_generated_files`
OUTPUT_DIR = os.getenv("MCP_OUTPUT_DIR", "../../agent_generated_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@mcp.tool(description="Plot top 10 RNA counts and summarize for a dataset from the OSDR unnormalized counts API.")
async def osdr_plot_top_rna(dataset_id: str) -> dict:
    """
    Fetches RNA count data for a given OSDR dataset, plots the top 10 expressed genes,
    saves the data and plot locally, and returns the file paths and a textual summary.
    """
    base_url = "https://visualization.osdr.nasa.gov/biodata/api/v2/query/data/"
    query_url = f"{base_url}?id.accession={dataset_id}&file.data%20type=unnormalized%20counts"

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(query_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data: {response.status_code}")

        try:
            df = pd.read_csv(StringIO(response.text))
        except Exception as e:
            preview = response.text[:300]
            raise RuntimeError(f"Could not parse CSV. Preview:\n{preview}") from e

    if df.empty:
        raise ValueError("No data returned from the API.")

    # File paths
    data_path = os.path.join(OUTPUT_DIR, f"{dataset_id}_unnormalized_counts.csv")
    plot_path = os.path.join(OUTPUT_DIR, f"{dataset_id}_top10_rna.png")
    summary_path = os.path.join(OUTPUT_DIR, f"{dataset_id}_top10_summary.txt")

    # Save raw data
    df.to_csv(data_path, index=False)

    # Calculate top genes
    gene_counts = df.drop(columns=[df.columns[0]]).sum(axis=1)
    top_genes = gene_counts.nlargest(10)
    top_gene_names = df.iloc[top_genes.index, 0]

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.bar(top_gene_names, top_genes)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total Count")
    plt.title(f"Top 10 Expressed Genes in {dataset_id}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Build and save summary
    summary_lines = [f"Top 10 expressed genes in study {dataset_id}:"]
    for name, count in zip(top_gene_names, top_genes):
        summary_lines.append(f"- {name}: {int(count):,} counts")
    summary_text = "\n".join(summary_lines)

    with open(summary_path, "w") as f:
        f.write(summary_text)

    return {
        "dataset_id": dataset_id,
        "summary": summary_text,
        "summary_file": os.path.abspath(summary_path),
        "data_file": os.path.abspath(data_path),
        "plot_file": os.path.abspath(plot_path),
    }

if __name__ == "__main__":
    mcp.run(transport='stdio')


# @mcp.tool(description="Plot top 10 RNA counts and summarize for a dataset from the OSDR unnormalized counts API.")
# async def osdr_plot_top_rna(dataset_id: str) -> dict:
#     """
#     Fetches RNA count data for a given OSDR dataset, plots the top 10 expressed genes,
#     saves the data and plot locally, and returns the file paths and a textual summary.
#     """
#     import httpx
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from io import StringIO
#     import os

#     base_url = "https://visualization.osdr.nasa.gov/biodata/api/v2/query/data/"
#     query_url = f"{base_url}?id.accession={dataset_id}&file.data%20type=unnormalized%20counts"

#     async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
#         response = await client.get(query_url)
#         if response.status_code != 200:
#             raise RuntimeError(f"Failed to fetch data: {response.status_code}")

#         try:
#             df = pd.read_csv(StringIO(response.text))
#         except Exception as e:
#             preview = response.text[:300]
#             raise RuntimeError(f"Could not parse CSV. Preview:\n{preview}") from e

#     if df.empty:
#         raise ValueError("No data returned from the API.")

#     # Save raw data
#     data_path = f"{dataset_id}_unnormalized_counts.csv"
#     df.to_csv(data_path, index=False)

#     # Calculate top genes
#     gene_counts = df.drop(columns=[df.columns[0]]).sum(axis=1)
#     top_genes = gene_counts.nlargest(10)
#     top_gene_names = df.iloc[top_genes.index, 0]

#     # Save plot
#     plt.figure(figsize=(10, 6))
#     plt.bar(top_gene_names, top_genes)
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel("Total Count")
#     plt.title(f"Top 10 Expressed Genes in {dataset_id}")
#     plt.tight_layout()
#     plot_path = f"{dataset_id}_top10_rna.png"
#     plt.savefig(plot_path)
#     plt.close()

#     # Build summary text
#     summary_lines = [
#         f"Top 10 expressed genes in study {dataset_id}:",
#     ]
#     for name, count in zip(top_gene_names, top_genes):
#         summary_lines.append(f"- {name}: {int(count):,} counts")

#     return {
#         "summary": "\n".join(summary_lines),
#         "data_file": os.path.abspath(data_path),
#         "plot_file": os.path.abspath(plot_path),
#     }



if __name__ == "__main__":
    mcp.run(transport='stdio')
    
