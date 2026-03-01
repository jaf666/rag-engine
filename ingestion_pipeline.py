import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

 
def load_files(data_path="data"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist")
    
    loader = DirectoryLoader(
        path=data_path,
        glob = "*.txt",
        loader_cls = TextLoader # Loader class to use for loading files
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {data_path}. Please add your documents")
    
    for i, doc in enumerate(documents[:2]):
        print(f"Document {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")  # Print the first 100 characters of the content
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
        
    return documents

def chunk_files(documents, chunk_size=800, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1}:")
            print(f"Content: {chunk.page_content[:100]}...")  # Print the first 100 characters of the chunk
            print(f"Content Length: {len(chunk.page_content)} characters")
            print(f"Metadata: {chunk.metadata}")
            print("-" * 50)
    
    return chunks


def create_vector_store(chunks):
    # Switching to a dedicated embedding model to ensure semantic accuracy and lower latency.
    # Generative models (like Gemma/Llama) are not optimized for vector transformations.
    embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                )
    vectorstore = InMemoryVectorStore.from_documents(
        chunks,
        embedding=embeddings,
    )       

    return vectorstore

def create_rag_chain(vectorstore):
    llm = OllamaLLM(model="gemma3")
    template = """ 
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Using LCEL to express query
    chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    print("")
    # Load the files
    test_files = load_files(data_path="./data")
    
    # Chunking the files
    chunks = chunk_files(test_files)
    
    # Embedding and Storing in Vector DB
    vector_store = create_vector_store(chunks)

    rag_chain = create_rag_chain(vector_store)

    question = "What is the main topic about SpaceX?"
    answer = rag_chain.invoke(question)

    print(f"Final annswer: {answer}")


if __name__ == "__main__":
    main()