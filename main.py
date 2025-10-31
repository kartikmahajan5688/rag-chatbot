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


# ---- 📄 Load PDF and TXT Documents ----
pdf_loader = PyPDFLoader("./documents/python-notes.pdf")
txt_loader = TextLoader("./documents/extra-notes.txt", encoding="utf-8")

pdf_docs = pdf_loader.load()
txt_docs = txt_loader.load()

# Combine both
documents = pdf_docs + txt_docs

# ---- ✂️ Split into Chunks ----
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Store in Pinecone
vectorstore = PineconeVectorStore.from_documents(
    docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME,
)

print("✅ PDF + TXT documents successfully stored in Pinecone!")

# Query test
query = "What topics are covered in these documents?"
results = vectorstore.similarity_search(query, k=3)
for i, res in enumerate(results, 1):
    print(f"\nResult {i}:\n{res.page_content[:300]}")
