import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from streamlit_chat import message
import tempfile
from llm import Loadllm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser

class PDFReader:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

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
        llm = Loadllm.load_llm()
 
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
