# OSDR MCP Server

This folder contains an MCP (Model Context Protocol) server that wraps simple, purpose-built tools around NASA's Open Science Data Repository (OSDR) API. These tools work with `mcp-agent` to prototype LLM-enabled workflows.

---

## 🧠 Project Goals

1. **Develop Visualization Tools**  
   Build reusable tools (e.g., PCA, heatmaps) that interface with LLMs supporting function calling for scientific visualization and interpretation.

2. **Enable Semantic Matching Workflows**  
   - Extract differential expression headers using the OSDR API  
   - Use metadata queries to find relevant samples matching specific biological conditions (e.g., FLT = spaceflight)  
   - Map condition strings back to actual sample names from the study

3. **Education & API Onboarding**  
   Provide clear, runnable examples that help researchers and developers learn how to work with the OSDR API through function-wrapped MCP tools.

---

## 🧰 Tech Stack

- [MCP](https://modelcontext.org/)
- Python (FastMCP)
- `mcp-agent` for orchestration
- NASA OSDR API

---

## 📁 Structure

- `agent_generated_files/`: Output directory for generated files (metadata, plots, etc.)
- `src/mcp_server_osdr/`: MCP server implementations for data fetching and visualization
- MCP tools are built using FastMCP and exposed over `stdio`

---

## 🚀 Getting Started

To run an MCP tool:
```bash
python main.py
