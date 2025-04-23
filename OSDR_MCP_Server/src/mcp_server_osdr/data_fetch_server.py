from mcp.server.fastmcp import FastMCP, Image
import httpx
import matplotlib.pyplot as plt
from io import StringIO
import os
import pandas as pd
import json

OUTPUT_DIR = os.getenv("MCP_OUTPUT_DIR", "../../agent_generated_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

mcp = FastMCP("osdr_data_fetch")

@mcp.tool(description="Fetch dataset metadata from the NASA OSDR API and save as JSON for downstream use.")
async def osdr_fetch_metadata(dataset_id: str) -> dict:
    """
    Fetch minimal metadata for a given dataset and save it to a local file.
    """
    url = f"https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/{dataset_id}/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    metadata = data.get(dataset_id, {}).get("metadata", {})

    cleaned = {
        "dataset_id": dataset_id,
        "title": metadata.get("study title", "N/A"),
        "organism": metadata.get("organism", "N/A"),
        "mission": ", ".join(metadata.get("mission", {}).get("name", [])) or "N/A",
        "protocols": metadata.get("study protocol name", []),
        "assay_type": metadata.get("study assay technology type", "N/A"),
        "platform": metadata.get("study assay technology platform", "N/A"),
        "funding": metadata.get("study funding agency", "N/A"),
    }

    # Save to JSON file
    filename = os.path.join(OUTPUT_DIR, f"{dataset_id}_metadata.json")
    with open(filename, "w") as f:
        json.dump(cleaned, f, indent=2)

    cleaned["metadata_file"] = os.path.abspath(filename)
    return cleaned

# @mcp.tool(description="Find OSDR studies accession numbers by organism name.")
# async def osdr_find_by_organism(organism_name: str) -> list[dict]:
#     """
#     Return a list of study IDs and organisms for a given organism name.
#     """
#     url = f"https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/*/metadata/organism/{organism_name}/"
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         response.raise_for_status()
#         data = response.json()

#     return [
#         {"dataset_id": dataset_id, "organism": details.get("metadata", {}).get("organism", "N/A")}
#         for dataset_id, details in data.items()
#     ]

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
    
