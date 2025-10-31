import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# print("🔑 OpenAI API Key:", OPENAI_API_KEY)
# print("🔑 Pinecone API Key:", PINECONE_API_KEY)
# print("📇 Pinecone Index Name:", PINECONE_INDEX_NAME)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if missing or dimension mismatch
indexes = {idx["name"]: idx for idx in pc.list_indexes()}

if PINECONE_INDEX_NAME not in indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    info = pc.describe_index(PINECONE_INDEX_NAME)
    if info.dimension != 1536:
        print("⚠️ Dimension mismatch — recreating index...")
        pc.delete_index(PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


# ---- Function to Load + Embed Documents ----
def ingest_documents(pdf_path=None, txt_path=None):
    loaders = []
    if pdf_path and os.path.exists(pdf_path):
        loaders.append(PyPDFLoader(pdf_path))
    if txt_path and os.path.exists(txt_path):
        loaders.append(TextLoader(txt_path, encoding="utf-8"))

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    if not documents:
        print("⚠️ No valid documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore.from_documents(
        chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME
    )

    print("✅ Documents successfully embedded and stored in Pinecone!")


if __name__ == "__main__":
    ingest_documents(
        pdf_path="./documents/python-notes.pdf",
        txt_path="./documents/extra-notes.txt",
    )
