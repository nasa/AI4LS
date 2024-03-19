import re
import requests

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from streamlit_chat import message
import tempfile
from llm import Loadllm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser

from langchain_core.pydantic_v1 import BaseModel, Field

from pydantic import BaseModel, Field

class WesternBlotProtocol(BaseModel):
    """Template for Western Blot protocol data."""
    
    sample_name: str = Field(..., description="Name of the sample.")
    protocol_ref: str = Field(..., description="Reference for the protocol used.")
    amount_of_protein_loaded: float = Field(..., description="Amount of protein loaded (µg).")
    type_of_gel: str = Field(..., description="Type of gel used in the process.")
    voltage: int = Field(..., description="Voltage (V) applied during the electrophoresis.")
    instrument_for_gel: str = Field(..., description="Instrument used for the gel electrophoresis.")
    type_of_transfer_membrane: str = Field(..., description="Type of transfer membrane used.")
    transfer_method: str = Field(..., description="Method used for the transfer.")
    blocking_chemical: str = Field(..., description="Chemical used for blocking.")
    blocking_concentration: str = Field(..., description="Concentration of the blocking chemical.")
    blocking_duration: str = Field(..., description="Duration of the blocking step.")
    blocking_temperature: str = Field(..., description="Temperature during the blocking step.")
    blocking_wash_buffer: str = Field(..., description="Wash buffer used after blocking.")
    blocking_wash_duration: str = Field(..., description="Duration of washing after blocking.")
    number_of_biological_markers: int = Field(..., description="Number of biological markers targeted.")
    marker_type: str = Field(..., description="Type of markers used.")
    protein_labeled: str = Field(..., description="Protein labeled (including any post-translational modification).")
    primary_chemical_used_for_dilution: str = Field(..., description="Chemical used for diluting the primary antibody.")
    primary_concentration: str = Field(..., description="Concentration of the primary antibody.")
    primary_antigen_host: str = Field(..., description="Host of the primary antigen.")
    primary_molecular_tag: str = Field(..., description="Molecular tag of the primary antibody (if unconjugated).")
    primary_fluorophore: str = Field(..., description="Fluorophore of the primary antibody (if conjugated).")
    primary_company_and_product_number: str = Field(..., description="Company and product number of the primary antibody.")
    primary_lot_number: str = Field(..., description="Lot number of the primary antibody.")
    primary_duration: str = Field(..., description="Duration of the primary antibody incubation.")
    primary_temperature: str = Field(..., description="Temperature during the primary antibody incubation.")
    primary_wash_buffer: str = Field(..., description="Wash buffer used after the primary antibody incubation.")
    primary_wash_duration: str = Field(..., description="Duration of washing after the primary antibody incubation.")
    secondary_chemical_used_for_dilution: str = Field(..., description="Chemical used for diluting the secondary antibody.")
    secondary_concentration: str = Field(..., description="Concentration of the secondary antibody.")
    secondary_antigen_host: str = Field(..., description="Host of the secondary antigen.")
    secondary_molecular_tag: str = Field(..., description="Molecular tag of the secondary antibody.")
    secondary_fluorophore: str = Field(..., description="Fluorophore of the secondary antibody.")
    secondary_company_and_product_number: str = Field(..., description="Company and product number of the secondary antibody.")
    secondary_lot_number: str = Field(..., description="Lot number of the secondary antibody.")
    secondary_duration: str = Field(..., description="Duration of the secondary antibody incubation.")
    secondary_temperature: str = Field(..., description="Temperature during the secondary antibody incubation.")
    secondary_wash_buffer: str = Field(..., description="Wash buffer used after the secondary antibody incubation.")
    secondary_wash_duration: str = Field(..., description="Duration of washing after the secondary antibody incubation.")
    imaging_substrate: str = Field(..., description="Imaging substrate used.")
    imaging_substrate_company_and_product_number: str = Field(..., description="Company and product number of the imaging substrate.")
    imaging_method: str = Field(..., description="Imaging method used.")
    exposure_time: str = Field(..., description="Exposure time for imaging.")
    quantification_stain: str = Field(None, description="Quantification stain used (optional).")
    quantification_software: str = Field(..., description="Software used for quantification.")
    lane_identification: str = Field(..., description="Identification of lanes.")
    number_of_lanes: int = Field(..., description="Number of lanes.")
    order_specified: str = Field(..., description="Order in which the samples were loaded.")
    protein_standard_or_ladder: str = Field(..., description="Protein standard or ladder used.")
    protein_standard_or_ladder_company_and_catalog_number: str = Field(..., description="Company and catalog number of the protein standard or ladder.")
# Continue defining fields as needed...

from typing import List, Optional


class ExtractionData(BaseModel):
    """Extracted information about key developments in the history of cars."""
    key_developments: List[WesternBlotProtocol]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found in the text.",
        ),
        # MessagesPlaceholder('examples'), # Keep on reading through this use case to see how to use examples to improve performance
        ("human", "{text}"),
    ]
)
llm = Loadllm.load_llm('mistral-7b-v0.1.Q4_K_M.gguf')


runnable = prompt | llm.with_structured_output(schema=WesternBlotProtocol)





class PDFReader:
    def __init__(self, uploaded_file, model):
        self.uploaded_file = uploaded_file
        self.model = model 
    def handlefileandingest(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        split_docs = text_splitter.split_documents(data)

        # Create embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create vectorstore from PDF
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
        
        #Load language model
        llm = Loadllm.load_llm(self.model)
 
        #Initial summary of PDF
        #template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. {context} Question: {question} Helpful Answer:"""
        #template = """<s> [INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context. Answer the question below from context below : {context} {question} [/INST] </s> """
        template = """[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. <</SYS>> \nAnswer the question from the following context. {context} \nQuestion: {question} [/INST]"""
        qa_prompt = PromptTemplate.from_template(template=template)
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        
        #rag_chain = (
        #    {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #    | qa_prompt
        #    | llm
        #    | StrOutputParser()
        #)

        #summary = rag_chain.invoke("Summarize the main themes of the study")

        # Create a conversational chain
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True,combine_docs_chain_kwargs={"prompt": qa_prompt}, verbose=True,)

        # Function for conversational chat
        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Feel free to ask me about the pdf."]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Uploaded " + self.uploaded_file.name]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Ask your question about the PDF", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    #message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    #message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), logo="https://osdr.nasa.gov/bio/images/logos/genelab_patch.png")
