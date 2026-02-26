import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

# load_dotenv()
 
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
    text_splitter = CharacterTextSplitter(
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



def main():
    print("")
    # Load the files
    test_files = load_files(data_path="./data")
    
    # Chunking the files
    chunks = chunk_files(test_files)
    # Embedding and Storing in Vector DB


if __name__ == "__main__":
    main()