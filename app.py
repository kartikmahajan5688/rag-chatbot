import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import gradio as gr
import uvicorn

# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ---- Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---- Initialize FastAPI ----
app = FastAPI(title="📚 RAG Chatbot with FastAPI + Pinecone + Gradio")

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

# ---- Chat Model ----
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3,
)


# ---- RAG Chat Function ----
def rag_chat(message, history):
    try:
        # Retrieve top documents
        results = vectorstore.similarity_search(message, k=3)
        context = "\n\n".join([r.page_content for r in results])

        # Construct the final prompt
        prompt = f"""
You are a helpful assistant. Use the following context to answer accurately.
If the context is not relevant, say "I’m not sure based on the provided documents."

Context:
{context}

User: {message}
Assistant:
"""

        # Generate response
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # ✅ IMPORTANT: Return only a string, not tuple/history
        return answer

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ---- Gradio Chat Interface ----
chat_interface = gr.ChatInterface(
    fn=rag_chat,
    title="📚 AI Document Chatbot (RAG + Pinecone + OpenAI)",
    description="Ask questions about your documents using Retrieval-Augmented Generation (RAG).",
    examples=[
        ["What topics are covered in the document?"],
        ["Explain Python decorators from the document."],
    ],
)


# ---- Mount Gradio App on FastAPI ----
@app.get("/")
def root():
    return {"message": "🚀 FastAPI + Gradio RAG Chatbot is running!"}


app = gr.mount_gradio_app(app, chat_interface, path="/gradio")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel
# from openai import OpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone

# # ---- Load environment variables ----
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # ---- Initialize Clients ----
# pc = Pinecone(api_key=PINECONE_API_KEY)
# client = OpenAI(api_key=OPENAI_API_KEY)

# # ---- Initialize Embeddings + Vector Store ----
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# vectorstore = PineconeVectorStore(
#     index_name=PINECONE_INDEX_NAME,
#     embedding=embeddings,
# )

# # ---- FastAPI App ----
# app = FastAPI(title="RAG using FastAPI + Pinecone + OpenAI")

# # ---- Request Schema ----


# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 3


# # ---- Root Endpoint ----
# @app.get("/")
# def home():
#     return {"message": "🚀 FastAPI RAG API is running successfully!"}


# # ---- RAG Endpoint ----
# @app.post("/query")
# def query_docs(request: QueryRequest):
#     # Step 1: Retrieve top-k similar documents
#     results = vectorstore.similarity_search(request.query, k=request.top_k)
#     context = "\n\n".join([r.page_content for r in results])

#     # Step 2: Build RAG prompt
#     prompt = f"""
# You are a knowledgeable AI assistant.
# Use the following context from company documents to answer the question accurately.
# If the answer cannot be found in the context, say "I’m not sure based on the available documents."

# Context:
# {context}

# Question: {request.query}
# Answer:
# """

#     # Step 3: Generate Answer with OpenAI
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",  # You can use gpt-4o or gpt-3.5-turbo
#         messages=[{"role": "user", "content": prompt}],
#     )

#     answer = completion.choices[0].message.content.strip()

#     # Step 4: Return the response
#     return {
#         "query": request.query,
#         "answer": answer,
#         "context_snippets": [r.page_content[:200] for r in results],
#     }
