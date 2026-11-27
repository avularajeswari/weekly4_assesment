
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "product_manual.pdf"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


def create_vectorstore():

    # Step 1: Load PDF
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Total PDF pages loaded: {len(documents)}")

    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_PATH),
        collection_name="product_manual"
    )

    print("Embeddings stored successfully in Chroma!")

    return vectorstore


if __name__ == "__main__":
    create_vectorstore()
