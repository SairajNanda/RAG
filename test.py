import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document  # Import Document
import time
from openai.error import RateLimitError, APIError

# Initialize Chroma client and embeddings
client = chromadb.Client()
embedding_model = OpenAIEmbeddings()
# Test with a small sample
sample_texts = ["AI stands for Artificial Intelligence.", "Python is a programming language."]
embeddings = embedding_model.embed_documents(sample_texts)
print(embeddings)
# Create a vector store to store document embeddings
doc_store = Chroma(client=client, embedding_function=embedding_model)

# Create a list of documents using the Document class
documents = [
    Document(id="1", page_content="AI stands for Artificial Intelligence."),
    Document(id="2", page_content="Python is a programming language."),
    Document(id="3", page_content="RAG combines generative models with retrieval-based methods."),
    Document(id="4", page_content="Chroma is used for vector storage and retrieval.")
]

# Add documents to Chroma store using the correct method
doc_store.add_documents(documents)

# Initialize OpenAI GPT model
llm = OpenAI(temperature=0.7)

# Create a Conversational RAG Chain (Retrieval-Augmented Generation)
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=doc_store.as_retriever()
)


def safe_add_documents(store, docs, retry_limit=5):
    for attempt in range(retry_limit):
        try:
            store.add_documents(docs)
            break  # Break loop if successful
        except (RateLimitError, APIError) as e:
            print(f"Error occurred: {e}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break



# Function for interactive chat with the chatbot
def chat_with_bot():
    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Get the response from the RAG system
        response = chat_chain.run(input=user_input)
        print(f"Chatbot: {response}")

# Start the chatbot
if __name__ == "__main__":
    chat_with_bot()
