import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings  # If used for embeddings
# import google.generativeai as genai  # If needed
from langchain_community.vectorstores import FAISS

# Assume a similar module exists for OpenAI in LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import os

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging an advanced model.
It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store,
and generates accurate answers to user queries. This advanced approach ensures high-quality,
contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. *Enter Your API Key*: You'll need an API key for the chatbot to access advanced models. Obtain your API key from the relevant platform.

2. *Upload Your Documents*: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. *Ask a Question*: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

# Access the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Vous êtes un assistant intéligent pour des étudiants à l'université. Répondez à la question aussi détaillée que possible en utilisant le contexte fourni. Assurez-vous de fournir tous les détails. Si la réponse n'est pas dans le contexte fourni, formulez gentiment une réponse en disant que vous ne savez pas.\n\n
    Contexte:\n {context}?\n
    Question: \n{question}\n

    Réponse:    
    """

    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.3, api_key=api_key)  # Adjusted to use OpenAI's model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.header("AI Document Assistant")

    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
        

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

    # Display existing questions and responses
    for i, (question, response) in enumerate(zip(st.session_state.questions, st.session_state.responses)):
        st.write(f"**Question {i+1}:** {question}")
        st.write(f"**Answer {i+1}:** {response}")

    # new_question = st.text_input(label="", placeholder="Ask a new question", value="", key="new_question")
    
    new_question = st.text_input(label="", placeholder="Ask a new question", value="", key="new_question")

    
    if st.button("Submit Question") and new_question and api_key:
        with st.spinner("Generating response..."):
            response = user_input(new_question, api_key)
            st.session_state.questions.append(new_question)
            st.session_state.responses.append(response)
            st.experimental_rerun()  # Rerun the app to display the new question and response
                
             
  
if __name__ == "__main__":
    main()
