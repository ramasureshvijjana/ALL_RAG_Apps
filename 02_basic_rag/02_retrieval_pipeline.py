from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load the Hugging Face API token from the .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the embedding model.
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the Chroma vector store with the embedding model and specify the persist directory and collection metadata.
db = Chroma(embedding_function = embed_model,
            persist_directory = "./02_basic_rag/VSDB/chroma",
            collection_metadata={"hnsw:space": "cosine"})

# Create a retriever from the Chroma vector store, specifying the number of similar documents to retrieve (k=5).
retriever = db.as_retriever(search_kwargs={"k": 5 })

# Process the query to retrieve similar documents from the vector store.
query = "What are the two main forms of Vitamin B3?"

similar_docs = retriever.invoke(query)

print(f"The Query : {query}")
for i, doc in enumerate(similar_docs):
    print(f"Similar document {i+1}: {doc.page_content}")

