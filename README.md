# 📚 AI Document Chatbot (RAG + Pinecone + OpenAI)

**Chat with your documents intelligently using Retrieval-Augmented Generation (RAG) powered by OpenAI, Pinecone, and FastAPI — with a beautiful Gradio UI.**

---

## 🚀 Features

✅ Upload and process **PDF or Text** documents
✅ Automatically **index content into Pinecone Vector Database**
✅ Chat with your document using **OpenAI GPT model (RAG pipeline)**
✅ **FastAPI backend** + **Gradio frontend** for easy deployment
✅ Supports **conversation memory** for contextual chatting
✅ Modular code for **scalable multi-document support**

---

## 🧠 Tech Stack

| Component           | Technology                                                     |
| ------------------- | -------------------------------------------------------------- |
| 💬 LLM              | [OpenAI GPT (via LangChain)](https://platform.openai.com/docs) |
| 🧩 Vector Store     | [Pinecone](https://www.pinecone.io/)                           |
| ⚙️ Framework        | [FastAPI](https://fastapi.tiangolo.com/)                       |
| 🧱 Frontend         | [Gradio](https://www.gradio.app/)                              |
| 🧰 Embeddings       | [LangChain OpenAIEmbeddings](https://python.langchain.com/)    |
| 📄 Document Loaders | LangChain Community Loaders                                    |
| ☁️ Deployment       | Uvicorn (ASGI Server)                                          |

---

## 📂 Project Structure

```
📁 ai-document-chatbot/
│
├── app.py                  # Main FastAPI + Gradio application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── documents/              # Your sample or uploaded documents
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kartikmahajan5688/rag-chatbot.git
cd ai-document-chatbot
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Create a `.env` File

Inside the project root, create a `.env` file and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-index
```

> 🧠 Make sure your Pinecone project is **serverless** and uses the same region as specified in the code (`us-east-1`).

---

## 🧾 Document Ingestion (Optional Step)

Before chatting, you can pre-load some documents manually using `main.py`.

```bash
python main.py
```

This will:

- Load PDFs or text files from the `./documents/` directory
- Split text into chunks
- Create embeddings using OpenAI
- Store them in Pinecone

---

## 💬 Run the Chatbot App

Start the FastAPI + Gradio interface:

```bash
python app.py
```

Once running, visit:

```
http://localhost:8000/gradio
```

You’ll see a beautiful Gradio UI like this:

```
📘 Upload your document and chat with it using AI!
```

1. Upload a PDF or Text file
2. Wait for it to process and index
3. Start chatting — ask natural language questions about your document!

---

## 🌐 API Access (Optional)

The FastAPI server is available under the same app.
If you deploy it (e.g., to Render or Railway), the Gradio UI mounts at `/gradio`.

---

## 🧩 Example Workflow

1. Upload a file like `company-policy.pdf`
2. The system:

   - Splits text into chunks
   - Creates embeddings
   - Stores in Pinecone
   - Enables semantic search retrieval

3. Chat:

   ```
   👤 User: What is the company’s leave policy?
   🤖 Bot: The company allows 24 paid leaves per year, as stated in section 3.2 of the document.
   ```

---

## 🧠 How It Works (Architecture)

```mermaid
flowchart TD
A[User Uploads Document] --> B[LangChain Loaders]
B --> C[Text Splitter (Chunks)]
C --> D[OpenAI Embeddings]
D --> E[Pinecone Vector Store]
E --> F[ConversationalRetrievalChain]
F --> G[ChatOpenAI (GPT)]
G --> H[Response via Gradio UI]
```

---

## 🧰 Requirements

| Library             | Version |
| ------------------- | ------- |
| fastapi             | latest  |
| uvicorn[standard]   | latest  |
| gradio              | latest  |
| python-dotenv       | latest  |
| pinecone-client     | latest  |
| langchain           | latest  |
| langchain-openai    | latest  |
| langchain-pinecone  | latest  |
| langchain-community | latest  |
| pypdf               | latest  |
| tiktoken            | latest  |

_(All managed via `requirements.txt`)_

---

## 🚀 Deployment Tips

- For **local use**, just run `python app.py`.
- To **deploy on Render / Railway / HuggingFace Spaces**, set:

  ```
  START_CMD = python app.py
  PORT = 8000
  ```

- Ensure environment variables (`.env`) are configured in the platform.

---

## 🧑‍💻 Author

**👋 Developed by:** _[Kartik Mahajan]_
📧 Email: [kartikmahajan5688@gmail.com](mailto:kartikmahajan5688@gmail.com)

---

## 🪪 License

This project is licensed under the **MIT License** – feel free to use, modify, and share!

---

## 💖 Support

If you found this project helpful, please ⭐ the repository and share it with others!
Let’s build smarter AI assistants together 🚀

---
