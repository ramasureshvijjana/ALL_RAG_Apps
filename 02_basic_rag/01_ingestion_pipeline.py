import os, sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load env variables and keys
load_dotenv()
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
print("=========================================\nThe env vars load done.")

# Load documents from data.
def load_documents(data_folder_path:str)->list:

    # Create a "directory loader" to load all text files from the specified folder.
    loader = DirectoryLoader(path=data_folder_path,
                             glob="*.txt",
                             show_progress=True,
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    
    # Load the documents using the loader.
    docs = loader.load()

    # Check if any documents were loaded and print the total count.
    if len(docs) == 0:
        print("No document found in the specified folder")
        return False
    else:
        print(f"Total {len(docs)} documents loaded successfully.{type(docs)}")

        # Print the content of the first few documents to verify they were loaded correctly.
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1} - type: {type(doc)}:\n")
            print(f"Content: {doc.page_content[:200]}")

        return docs
    
# Split the loaded documents into smaller chunks.
def chunk_documents(docs:list, chunk_size:int=200, chunk_overlap:int = 50)->list:

    # Create a RecursiveCharacterTextSplitter to split the documents into smaller chunks.
    r_text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    # Split the documents into chunks using the text splitter
    chunked_docs = r_text_splitter.split_documents(docs)

    # # Print the content of few chunk
    for i, chunk in enumerate(chunked_docs):
        print(f"\nChunk {i+1}\n=======================================")
        print(f"Content: {chunk.page_content[:200]}")
        print("=======================================")

        if i == 3: break
        
    return chunked_docs


def embed_documents(chunked_docs:list, vsdb_path:str = "./02_basic_rag/VSDB/chroma"):

    # Loading the embedding model from huggingface
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Creating Chroma Vector store DB.
    print("Creating Chroma Vector store DB.")
    chroma_db = Chroma.from_documents(documents=chunked_docs, 
                                      embedding=embedding_model, 
                                      persist_directory=vsdb_path,
                                      collection_metadata={"hnsw:space": "cosine"})
    
    return chroma_db

    

def main(data_folder_path):
    docs = load_documents(data_folder_path)
    chunked_docs = chunk_documents(docs)
    chroma_db = embed_documents(chunked_docs, vsdb_path="./02_basic_rag/VSDB/chroma")
    print("Ingestion pipeline executed successfully.")


if __name__ == "__main__":
    data_folder_path = sys.argv[1] 
    main(data_folder_path)