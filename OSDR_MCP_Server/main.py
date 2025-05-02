import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM

app = MCPApp(name="osdr_parallel_workflow")


async def example_usage():
    async with app.run() as parallel_app:
        logger = parallel_app.logger
        context = parallel_app.context
        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory for filesystem access
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        dataset_id = "OSD-120"

        # Agent 1: metadata fetcher
        metadata_agent = Agent(
            name="metadata_agent",
            instruction=f"""
            Fetch and save metadata for dataset {dataset_id}.
            Use the appropriate tool to store the metadata to a JSON file.
            Do not summarize the contents.
            """,
            # instruction="""
            # You specialize in retrieving metadata for a given OSDR dataset ID.
            # Use the available tools to fetch study title, organism, assay type, mission, and other key metadata.
            # You do not summarize or analyze the data — just retrieve and structure it.
            # """,
            server_names=["osdr_data_fetch", "filesystem"],
        )

        # Agent 2: RNA quant analysis
        quant_analysis_agent = Agent(
            name="quant_analysis_agent",
            # instruction="""
            # You are responsible for analyzing RNA-seq data from OSDR studies.
            # Use the appropriate tool to generate a bar plot of the top 10 expressed genes for a given dataset.
            # Return the visualization file, path to the raw data, and a textual summary of the top genes.
            # """,
            instruction=f"""
            Analyze the unnormalized RNA-seq data for dataset {dataset_id}.
            Generate a bar plot of the top 10 expressed genes.""",
            #   and save it to file.
            # Also save a short summary as a .txt file for later synthesis.
            # """,
            server_names=["osdr_viz_tools"],
        )

        # Agent 3: summary writer
        summary_writer_agent = Agent(
            name="summary_writer_agent",
            instruction=f"""
            You will receive:
            1. A {dataset_id}_metadata.json file containing metadata
            2. A "{dataset_id}_top10_summary.txt file summarizing top 10 RNA genes

            Combine the metadata and RNA insights into a clean, human-readable 1-page markdown summary report.
            Save it as '{dataset_id}_summary.md'.
            """,
            # instruction="""
            # You synthesize findings from the other agents to produce a clear, 1-page summary report.
            # Combine metadata from the metadata_agent with top RNA gene results from the quant_analysis_agent.
            # Format your output clearly for humans and save the result to disk.
            # """,
            server_names=["filesystem"],
        )

        # Set up parallel fan-out and fan-in
        parallel = ParallelLLM(
            fan_in_agent=summary_writer_agent,
            fan_out_agents=[metadata_agent, quant_analysis_agent],
            llm_factory=OpenAIAugmentedLLM,
        )

        # task = """
        # Use dataset ID 'OSD-120' from the NASA OSDR database.
        # 1. Retrieve the study metadata, including title, organism, assay type, mission, and funding agency.
        # 2. Generate a bar plot of the top 10 expressed genes from the RNA-seq unnormalized counts data.
        # 3. Summarize both sets of findings into a 1-page markdown report and save as 'OSD-120_summary.md'.
        # """

        # Run the fan-out/fan-in workflow
        result = await parallel.generate_str(
            # message=task,
            message=f"OSDR study {dataset_id}",
            request_params=RequestParams(model="llama3.2"),
        )

        logger.info(f"Final result:\n{result}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    print(f"Total run time: {end - start:.2f}s")



# import asyncio
# import os

# from mcp_agent.app import MCPApp
# from mcp_agent.agents.agent import Agent
# from mcp_agent.workflows.llm.augmented_llm import RequestParams
# from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
# from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator

# app = MCPApp(name="osdr_parallel_workflow")


# async def example_usage():
#     async with app.run() as orchestrator_app:
#         logger = orchestrator_app.logger
#         context = orchestrator_app.context

#         logger.info("Current config:", data=context.config.model_dump())

#         # Add current directory for any write actions
#         context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

#         # Agents
#         metadata_agent = Agent(
#             name="metadata_agent",
#             instruction="""
#             You specialize in retrieving metadata for a given OSDR dataset ID.
#             Use the available tools to fetch study title, organism, assay type, mission, and other key metadata.
#             You do not summarize or analyze the data — just retrieve and structure it.
#             """,
#             server_names=["osdr_data_fetch", "filesystem"]
#         )

#         # Agent 2: Extracts and visualizes RNA count data
#         quant_analysis_agent = Agent(
#             name="quant_analysis_agent",
#             instruction="""
#             You are responsible for analyzing RNA-seq data from OSDR studies.
#             Use the appropriate tool to generate a bar plot of the top 10 expressed genes for a given dataset.
#             Return the visualization file, path to the raw data, and a textual summary of the top genes.
#             """,
#             server_names=["osdr_data_fetch", "filesystem"]
#         )

#         # Agent 3: Summarizes the findings from the other agents
#         summary_writer_agent = Agent(
#             name="summary_writer_agent",
#             instruction="""
#             You synthesize findings from other agents to produce a clear, 1-page summary report.
#             Combine metadata from the metadata_agent with top RNA gene results from the quant_analysis_agent.
#             Format your output clearly for humans and save the result to disk.
#             """,
#             server_names=["filesystem"]
#         )

#         # # Task prompt for orchestrator
#         # task = """
#         # Use dataset ID 'OSD-120' from the NASA OSDR database.

#         # 1. Retrieve and summarize the study metadata, including title, organism, assay type, mission, and funding agency.
#         # 2. Generate a bar plot of the top 10 expressed genes from the RNA-seq unnormalized counts data.
#         # 3. Write a clear 1-page markdown report combining both the metadata summary and insights from the gene expression data.
#         # 4. Save the final report to disk as 'OSD-120_summary.md'.
#         # """

#         # Instantiate the orchestrator with updated agent roles
#         orchestrator = Orchestrator(
#             llm_factory=OpenAIAugmentedLLM,
#             available_agents=[
#                 metadata_agent,
#                 quant_analysis_agent,
#                 summary_writer_agent,
#             ],
#             plan_type="iterative",
#         )

#         # Run the orchestrator task
#         result = await orchestrator.generate_str(
#             message=task,
#             request_params=RequestParams(model="llama3.2")
#         )

#         logger.info(f"Final result:\n{result}")



# if __name__ == "__main__":
#     import time
#     start = time.time()
#     asyncio.run(example_usage())
#     end = time.time()
#     print(f"Total run time: {end - start:.2f}s")

