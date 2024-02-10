# Streamlit Chatbot with RAG Model for PDF and OSDR Study Parsing

## Overview
This project introduces a cutting-edge chatbot feature, implemented using Streamlit and a Retriever-Reader (RAG) model from LangChain. The chatbot has two primary capabilities:
1. **Ingesting PDF documents:** Enabling users to ask questions about their content.
2. **Parsing OSDR studies:** Specifically designed for NASA's Open Science Data Repository (OSDR), the chatbot can respond to queries about these studies.

## Features
- **RAG Model Integration:** Employs LangChain's RAG model for efficient information retrieval and accurate question answering based on ingested content.
- **PDF Ingestion and Query Handling:** Users can upload PDFs for processing and querying via the chatbot interface. This leverages the RAG model's capabilities to understand and respond to content-related questions.
- **OSDR Study Parsing:** Tailored for parsing OSDR studies, facilitating users, especially researchers, to gain insights from these comprehensive documents.
- **Query Memory:** The chatbot is adept at remembering previous queries within a session, enabling context-aware interactions and continuous conversation flow.
- **Flexible Model Integration:** Utilizes the open-source LLM trained model (llama-2-13b-chat.Q4_K_M.gguf), with a design that supports easy integration with other LLM models as per evolving needs.
- **Streamlit Interface:** Streamlit's framework is used to create a user-friendly and intuitive chatbot interface.

## Testing
- Rigorous testing has been conducted with various PDF documents and OSDR studies to ensure the chatbot's accuracy and reliability.
- **How to Test:**
  - Launch the Streamlit app.
  - Upload a PDF document and ask questions pertinent to its content.
  - Engage with the OSDR study mode to query about specific pre-loaded studies.

## Impact
This feature significantly bolsters the application's ability to process and extract information from complex documents, rendering it an invaluable tool for researchers and general users dealing with PDFs and OSDR studies.

## Screenshots
![Chatbot Interface](https://github.com/nasa/AI4LS/blob/osdr-chatbot-integration/chatbot/icons/294312948-5ec7036a-8a05-419a-b587-0b124273c90c.png?raw=true)
*Figure 1: Streamlit Chatbot interface demonstrating PDF ingestion and query handling.*

## Additional Notes
- Future updates will focus on enhancing text extraction methods for improved vector matching and result accuracy.
