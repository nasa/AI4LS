import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader, AsyncHtmlLoader, PlaywrightURLLoader, SeleniumURLLoader, TextLoader, UnstructuredHTMLLoader, BSHTMLLoader
from streamlit_chat import message
import tempfile
from llm import Loadllm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain.schema import StrOutputParser
import chromedriver_autoinstaller
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
#chromedriver_autoinstaller.install()

class WebReader:
    def __init__(self, uploaded_url):
        self.uploaded_url = uploaded_url
    
    def loadweb(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        url = self.uploaded_url
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=chrome_options)

        driver.get(url) 
        html = driver.page_source

        # this renders the JS code and stores all
        # of the information in static HTML code.
        
        # Now, we could simply apply bs4 to html variable
        soup = BeautifulSoup(html, "html.parser") 
        driver.quit()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            # Write the string to the temporary file
            temp_file.write(soup.get_text())
            # Get the name of the file
            temp_file_name = temp_file.name

        loader = TextLoader(temp_file_name)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        split_docs = text_splitter.split_documents(data)

        # Create embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create vectorstore from PDF
        vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        
        #Load language model
        llm = Loadllm.load_llm()
 
        #Initial summary of PDF
        #template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. {context} Question: {question} Helpful Answer:"""
        #template = """<s> [INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context. Answer the question below from context below : {context} {question} [/INST] </s> """
        template = """[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> Answer the question from the following context. {context} Question: {question} [/INST]"""
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
            st.session_state['generated'] = ["Hello! Feel free to ask me about the study."]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["OSDR Study: " + self.uploaded_url]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Ask your question about the experiment", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), logo="https://osdr.nasa.gov/bio/images/logos/genelab_patch.png")
