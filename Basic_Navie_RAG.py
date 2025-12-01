# --- Generator / Retrieval + LLM response stage ---

# 1) imports (add these to top with your existing imports)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.llms import Ollama   # Ollama LLM wrapper

# 1. RAG 3 Stages: Document Loading, Splitting, Embedding
text_doc = TextLoader(r"..\01_data_ingestion\speech.txt").load()
splitted_docs = RecursiveCharacterTextSplitter(chunk_size= 1000, 
                                               chunk_overlap=20).split_documents(text_doc)
ollama_embedder = OllamaEmbeddings(model = 'llama3.2:1b')

# 2. Creating FAISS Vector Store DB:
faiss_db = FAISS.from_documents(splitted_docs, ollama_embedder)
input_text = "We have borne with their present government."

## ----- DEBUG -----
most_matched_docs = faiss_db.similarity_search(input_text)
print(f"""----------------------------------------------------------
      The most matched docs of input query : '{input_text}' is : 
      ---------------------------------------------------------- 
      
      {most_matched_docs[0].page_content}
      
      ----------------------------------------------------------""")

# We can search the query with score. It will desplay the score also.
docs_and_score=faiss_db.similarity_search_with_score(input_text)
docs_and_score

# 3. Similarity Search with Vector
embedding_vector=ollama_embedder.embed_query(input_text)
searched_docs=faiss_db.similarity_search_by_vector(embedding_vector)
searched_docs

# 4. Converting FAISS DB as a Retriever
retriever=faiss_db.as_retriever()
docs=retriever.invoke(input_text)
docs[0].page_content

### Saving And Loading
faiss_db.save_local("faiss_DB")
new_db=FAISS.load_local("faiss_DB",
                        ollama_embedder,
                        allow_dangerous_deserialization=True)

# 2) create a retriever from your FAISS DB (tweak k/search params as desired)
retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 3) instantiate the Ollama LLM you'll use to generate answers
#    adjust temperature, max_tokens, or model name as needed
llm = Ollama(model="llama3.2:1b", temperature=0.0)  # deterministic answers with temp=0.0

# -----------------------
# Option A — simple RetrievalQA (one-shot QA)
# -----------------------
# You can customize the prompt template if you want the LLM to behave in a certain way.
prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "You are a helpful assistant. Use the provided context to answer the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer concisely and reference the context when relevant."
    )
)

# ----------------------------------------------
# 5) Document Chain (combine retrieved docs)
# ----------------------------------------------
document_chain = create_stuff_documents_chain(llm, prompt)

# ----------------------------------------------
# 6) Full Retrieval + Generation Chain
# ----------------------------------------------
rag_chain = create_retrieval_chain(retriever, document_chain)

# ----------------------------------------------
# 7) Ask a question
# ----------------------------------------------
query = "We have borne with their present government."
result = rag_chain.invoke({"question": query})

print("\n================== FINAL ANSWER ==================\n")
print(result["answer"])

print("\n================== USED DOCUMENTS ==================\n")
for i, doc in enumerate(result["context"][:3]):
    print(f"[Doc-{i}] {doc.page_content[:300]}\n")



# -----------------------
# Notes & tips
# -----------------------
# - If documents are large and you want better scaling, try chain_type='map_reduce' or 'refine'.
# - To tune result length, use LLM args like `max_tokens` if supported by the Ollama wrapper.
# - For deterministic answers set temperature=0.0; increase for more creative responses.
# - If you want to customize retrieval (e.g., use BM25-like weighting or metadata filters), set retriever.search_kwargs or pass metadata filters into .as_retriever() depending on your wrapper.
# - If you saved & loaded the FAISS DB (`FAISS.load_local(...)`) simply repeat:
#       new_db = FAISS.load_local("faiss_DB", ollama_embedder, allow_dangerous_deserialization=True)
#       retriever = new_db.as_retriever(...)
