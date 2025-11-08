import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import gradio as gr
import uvicorn
from langchain.prompts import PromptTemplate
import hashlib


# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

current_index_name = os.getenv("PINECONE_INDEX_NAME")  # default on startup

# ---- Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---- Initialize FastAPI ----
app = FastAPI(title="📚 RAG Chatbot with FastAPI + Gradio + Pinecone")

# ---- Allow CORS for Gradio ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Setup Embeddings + Vectorstore ----
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME, embedding=embeddings)

# ---- Chat Model + Memory ----
llm = ChatOpenAI(model="gpt-4o-mini",
                 openai_api_key=OPENAI_API_KEY, temperature=0.3)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

# ---- Build Conversational Retrieval QA Chain ----
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    verbose=True,
)

QA_PROMPT = PromptTemplate.from_template("""
You are an intelligent document assistant. Use ONLY the context provided below to answer.
If the answer is not in the context, politely say you don’t know.

Context:
{context}

Question: {question}
""")

qa_chain.combine_docs_chain.llm_chain.prompt = QA_PROMPT

# ---- Function: Handle document upload ----


print("qa_chain", qa_chain)
print("vectorstore", vectorstore)
print("current_index_name", current_index_name)


def upload_document(file):
    global qa_chain, vectorstore, current_index_name, memory

    try:
        if file is None:
            return "⚠️ Please upload a file first."

        # Gradio gives a NamedString with .name pointing to temp path
        temp_path = file.name
        if not temp_path or not os.path.exists(temp_path):
            return "⚠️ Could not locate uploaded file path."

        # ---- Load the file ----
        if temp_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")

        docs = loader.load()
        if not docs:
            return "⚠️ No text found in the uploaded file."

        # ---- Split into chunks ----
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # ---- Create a valid Pinecone index name ----
        base_name = os.path.splitext(os.path.basename(file.name))[0].lower()
        # Replace invalid chars with '-'
        sanitized_name = "".join(
            c if c.isalnum() or c == "-" else "-" for c in base_name)
        safe_hash = hashlib.sha1(base_name.encode()).hexdigest()[:8]
        index_name = f"rag-{sanitized_name[:25]}-{safe_hash}"

        # ---- Check existing indexes ----
        existing = (
            [i["name"] for i in pc.list_indexes()]
            if isinstance(pc.list_indexes(), list)
            else [i["name"] for i in pc.list_indexes().get("indexes", [])]
        )

        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )

        # ---- Upload embeddings ----
        PineconeVectorStore.from_documents(
            chunks, embedding=embeddings, index_name=index_name)

        # ---- Update retriever & chain ----
        current_index_name = index_name
        vectorstore = PineconeVectorStore(
            index_name=current_index_name, embedding=embeddings)

        # Reset conversation memory for new file
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True,
            verbose=True,
        )

        qa_chain.combine_docs_chain.llm_chain.prompt = QA_PROMPT

        return f"✅ Uploaded and indexed '{file.name}' successfully! (index: {index_name})"

    except Exception as e:
        return f"⚠️ Error uploading file: {e}"


# ---- Function: Handle chat ----
def chat_with_doc(message, history):
    try:
        # Ensure qa_chain exists
        if qa_chain is None:
            return "⚠️ QA chain not initialized."

        # Invoke chain
        result = qa_chain.invoke({"question": message})

        # Debug - show which index / snippets were used
        # The retriever will fetch from current_index_name now
        src_docs = result.get("source_documents", [])
        snippets = [d.page_content[:200].replace("\n", " ") for d in src_docs]
        print("🔎 current_index_name:", current_index_name)
        print("🔍 Retrieved snippets:", snippets)

        return result.get("answer", "⚠️ No answer returned.")
    except Exception as e:
        return f"⚠️ Error: {e}"


# ---- Gradio Interface ----
with gr.Blocks(title="📚 AI Document Chatbot (RAG + Pinecone + OpenAI)") as demo:
    gr.Markdown("### 📘 Upload your document and chat with it using AI!")

    with gr.Row():
        file_input = gr.File(label="Upload a PDF or Text file")
        upload_button = gr.Button("📤 Process Document")

    upload_output = gr.Textbox(label="Upload Status")

    upload_button.click(upload_document, inputs=file_input,
                        outputs=upload_output)

    gr.Markdown("### 💬 Chat with your uploaded document")
    chatbot = gr.ChatInterface(
        fn=chat_with_doc,
        title="Document Chatbot",
        description="Ask questions about the uploaded document.",
    )


# ---- Mount Gradio on FastAPI ----
app = gr.mount_gradio_app(app, demo, path="/gradio")


# ---- Run ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
