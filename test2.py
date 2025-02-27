import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import streamlit as st
import os

# Step 1: Initialize Chroma client and OpenAI embeddings model
client = chromadb.Client()  # Chroma client to interact with the vector database
embedding_model = OpenAIEmbeddings()  # OpenAI model to convert text to embeddings

# Step 2: Create a vector store (Chroma) to store document embeddings
doc_store = Chroma(client=client, embedding_function=embedding_model)

# Step 3: Define a few example documents that the chatbot can use
documents = [
    Document(id="1", page_content="AI stands for Artificial Intelligence. It is a branch of computer science."),
    Document(id="2", page_content="Python is a high-level programming language that is widely used."),
    Document(id="3", page_content="Retrieval-Augmented Generation (RAG) is a method that combines search with generative models like GPT-3."),
    Document(id="4", page_content="Chroma is a database used to store and retrieve document embeddings for fast search."),
]

# Step 4: Add documents to the Chroma vector store
doc_store.add_documents(documents)

# Step 5: Initialize OpenAI model for conversation generation
llm = OpenAI(temperature=0.7)  # You can adjust temperature for randomness

# Step 6: Create a Conversational Retrieval Chain using the retriever and LLM
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=doc_store.as_retriever()
)

# Step 7: Create a function for interactive chat
def chat_with_bot(user_input):
    # Get the response from the RAG system (retrieve and generate response)
    response = chat_chain.run(input=user_input)
    return response

# Step 8: Streamlit UI to interact with the chatbot
st.title("AI Chatbot with RAG")
st.write("Type your message below to chat with the bot. Type 'exit' to end the chat.")

# Input box for user input
user_input = st.text_input("You: ")

if user_input:
    if user_input.lower() == "exit":
        st.write("Goodbye!")
    else:
        # Display the chatbot's response
        response = chat_with_bot(user_input)
        st.write(f"Chatbot: {response}")
