# 🏥 MediBot - AI-Powered Medical Chatbot

An intelligent medical information chatbot powered by Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) technology. MediBot provides accurate medical information by leveraging the GALE Encyclopedia of Medicine as its knowledge base.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Keys Setup](#api-keys-setup)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

MediBot is a Retrieval-Augmented Generation (RAG) chatbot designed to answer medical queries using authoritative medical literature. The system combines the power of vector databases, semantic search, and large language models to provide contextually relevant and accurate medical information.

**⚠️ Disclaimer**: This chatbot is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.

## ✨ Features

- 🤖 **Intelligent Query Processing**: Uses advanced NLP to understand medical queries
- 📚 **Knowledge Base**: Powered by The GALE Encyclopedia of Medicine
- 🎯 **Context-Aware Responses**: RAG architecture ensures responses are grounded in medical literature
- 🔍 **Source Attribution**: Shows source documents for transparency and verification
- 💬 **Interactive UI**: User-friendly Streamlit interface with chat history
- ⚡ **Fast Retrieval**: FAISS vector database for efficient semantic search
- 🔒 **Secure**: Environment-based API key management
- 🌐 **Flexible LLM Support**: Works with Groq (free tier) and HuggingFace models

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Streamlit Web App)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PROCESSING                           │
│                   (LangChain RetrievalQA)                       │
└────────┬────────────────────────────────────────────┬───────────┘
         │                                            │
         ▼                                            ▼
┌────────────────────────┐              ┌─────────────────────────┐
│   VECTOR RETRIEVAL     │              │    LLM GENERATION       │
│   (FAISS Database)     │              │  (Groq/HuggingFace)    │
│                        │              │                         │
│  - Semantic Search     │◄────────────►│  - Context Analysis     │
│  - Top-k Documents     │              │  - Answer Generation    │
│  - Embeddings Model    │              │  - Source Attribution   │
└────────────────────────┘              └─────────────────────────┘
         │                                            │
         ▼                                            ▼
┌────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE                              │
│           (GALE Encyclopedia of Medicine - PDF)                │
│                                                                │
│  Raw PDF → Chunks → Embeddings → FAISS Index                  │
└────────────────────────────────────────────────────────────────┘
```

### Architecture Flow:

1. **Data Ingestion Pipeline**:
   - PDF documents loaded from `/data` directory
   - Text extracted and split into manageable chunks (500 chars with 50 char overlap)
   - Embeddings generated using Sentence Transformers
   - Stored in FAISS vector database for fast retrieval

2. **Query Processing**:
   - User query → Embedding generation
   - Semantic search in FAISS (retrieves top-3 relevant chunks)
   - Context + Query sent to LLM

3. **Response Generation**:
   - LLM generates answer based on retrieved context
   - Custom prompt ensures grounded responses
   - Source documents returned for verification

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.9+**: Programming language
- **LangChain**: Framework for LLM application development
- **Streamlit**: Web interface framework
- **FAISS**: Vector database for similarity search
- **HuggingFace**: Embeddings and LLM models
- **Groq**: Fast LLM inference API

### Key Libraries
- `langchain-community`: Community integrations
- `langchain-groq`: Groq LLM integration
- `langchain-huggingface`: HuggingFace models
- `sentence-transformers`: Text embeddings
- `pypdf`: PDF processing
- `faiss-cpu`: Vector similarity search
- `python-dotenv`: Environment management

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip or pipenv
- Git

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/rizwimohdaltamash/Medibot.git
cd Medibot/medical-chatbot-main
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

Using pip:
```bash
pip install -r requirements.txt
```

Or using pipenv:
```bash
pipenv install
pipenv shell
```

4. **Set up environment variables**
```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

Add your API keys to `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for HuggingFace models
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM inference | Yes (for Groq model) |
| `HF_TOKEN` | HuggingFace token for model access | Optional |

### Model Configuration

The project supports multiple LLM backends:

**Groq (Default - Free Tier)**:
```python
llm = ChatGroq(
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.0,
    groq_api_key=os.environ["GROQ_API_KEY"]
)
```

**HuggingFace** (Commented out in code):
```python
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    model_kwargs={"token": HF_TOKEN, "max_length": "512"}
)
```

## 🚀 Usage

### 1. Create Vector Database (First Time Only)

```bash
python create_memory_for_llm.py
```

This script:
- Loads PDF files from `/data` directory
- Splits text into chunks
- Generates embeddings
- Creates and saves FAISS index to `/vectorstore/db_faiss`

### 2. Run the Chatbot (Streamlit App)

```bash
streamlit run medibot.py
```

The web app will open at `http://localhost:8501`

### 3. Test with CLI (Alternative)

```bash
python connect_memory_with_llm.py
```

Interactive command-line interface for testing queries.

### Sample Queries

- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What causes asthma?"
- "Explain the difference between Type 1 and Type 2 diabetes"

## 📁 Project Structure

```
medical-chatbot-main/
│
├── medibot.py                      # Main Streamlit application
├── create_memory_for_llm.py        # Vector database creation script
├── connect_memory_with_llm.py      # CLI testing interface
│
├── data/                           # Medical knowledge base
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
│
├── vectorstore/                    # Vector database storage
│   └── db_faiss/
│       ├── index.faiss            # FAISS index file
│       └── index.pkl              # Metadata pickle file
│
├── requirements.txt               # Python dependencies
├── Pipfile                        # Pipenv configuration
├── Pipfile.lock                   # Locked dependencies
└── README.md                      # Project documentation
```

## 🔬 How It Works

### 1. Document Processing (`create_memory_for_llm.py`)

```python
# Load PDFs
documents = DirectoryLoader("data/", glob='*.pdf', loader_cls=PyPDFLoader).load()

# Create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Generate embeddings and store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore/db_faiss")
```

### 2. Query Processing (`medibot.py`)

```python
# Load vector database
db = FAISS.load_local("vectorstore/db_faiss", embeddings)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(...),
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': custom_prompt}
)

# Process query
response = qa_chain.invoke({'query': user_question})
```

### 3. Custom Prompt Template

The system uses a carefully crafted prompt to ensure accurate, grounded responses:

```
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
```

## 🔑 API Keys Setup

### Groq API (Recommended - Free Tier)

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create new API key
5. Add to `.env` file

### HuggingFace Token (Optional)

1. Visit [HuggingFace](https://huggingface.co/)
2. Create account and go to Settings → Access Tokens
3. Generate new token with read permissions
4. Add to `.env` file

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **The GALE Encyclopedia of Medicine** for medical knowledge base
- **LangChain** for the RAG framework
- **HuggingFace** for embeddings models
- **Groq** for fast LLM inference
- **Streamlit** for the web interface

## 📧 Contact

Mohd. Altamash Rizwi - [@rizwimohdaltamash](https://github.com/rizwimohdaltamash)

Project Link: [https://github.com/rizwimohdaltamash/Medibot](https://github.com/rizwimohdaltamash/Medibot)

---

**⚠️ Important Medical Disclaimer**: This chatbot provides general medical information for educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
