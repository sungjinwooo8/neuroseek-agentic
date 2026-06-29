# 🤖 AI Blog Search - Agentic RAG Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Intelligent document search powered by Agentic RAG & LangGraph**

[🌐 Live Demo](https://sungjinwooo8-neuroseek-agentic-app-rvyxvc.streamlit.app/) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

AI Blog Search is a sophisticated Retrieval-Augmented Generation (RAG) application that enables intelligent question-answering over web documents. Built with **LangGraph** and **LangChain**, it uses an agentic workflow to retrieve, evaluate, and generate accurate responses from indexed blog content.

> 🌐 **Try it now**: [Live Demo](https://sungjinwooo8-neuroseek-agentic-app-rvyxvc.streamlit.app/)

### Key Highlights

- 🧠 **Agentic RAG**: Uses LangGraph to orchestrate intelligent retrieval and generation workflows
- 🔍 **Smart Retrieval**: Automatically evaluates document relevance and rewrites queries when needed
- 💾 **Local Storage**: Uses ChromaDB for free, persistent vector storage
- 🎨 **Modern UI**: Beautiful Streamlit interface with gradient designs
- ⚡ **Fast & Efficient**: Optimized document processing and retrieval

---

## ✨ Features

### Core Capabilities

- **📄 Document Indexing**: Add any blog URL to create a searchable knowledge base
- **🔎 Intelligent Search**: Ask natural language questions about indexed content
- **🤖 Agentic Workflow**: 
  - Automatic query understanding
  - Document relevance scoring
  - Query rewriting for better results
  - Context-aware answer generation
- **💬 Natural Language Interface**: Ask questions in plain English
- **🔄 Persistent Storage**: Documents are stored locally and persist between sessions

### Technical Features

- **Vector Embeddings**: Uses Google Gemini embeddings for semantic search
- **Chunking Strategy**: Intelligent text splitting with overlap for context preservation
- **Relevance Grading**: LLM-powered document relevance assessment
- **Query Transformation**: Automatic query improvement for better retrieval

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Agent Node    │ ──► Decides to retrieve or end
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retrieve Node  │ ──► Searches vector database
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grade Node     │ ──► Evaluates document relevance
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Generate│ │ Rewrite  │ ──► Improves query if docs irrelevant
└────────┘ └──────────┘
    │
    ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

### Components

1. **Agent Node**: Decides whether to retrieve documents or end the workflow
2. **Retrieve Node**: Searches the ChromaDB vector store using semantic similarity
3. **Grade Node**: Evaluates if retrieved documents are relevant to the query
4. **Rewrite Node**: Transforms queries to improve retrieval when documents are irrelevant
5. **Generate Node**: Creates final answers using retrieved context

---

## 📦 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed on your system
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **pip** package manager

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd rag
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import langchain; print('Installation successful!')"
```

---

## ⚙️ Configuration

### API Key Setup

1. **Get your Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key
   - Copy the API key

2. **Configure in the App**:
   - Run the application (see Usage section)
   - Open the sidebar
   - Enter your Gemini API key
   - Click "Save API Key"

> ⚠️ **Security Note**: Never commit your API keys to version control. Consider using environment variables for production deployments.

---

## 💻 Usage

### Option 1: Use the Live Demo

🌐 **[Access the deployed application](https://sungjinwooo8-neuroseek-agentic-app-rvyxvc.streamlit.app/)** - No installation required!

Simply visit the link above, enter your Gemini API key, and start using the application.

### Option 2: Run Locally

```bash
streamlit run app.py
```

The application will open in your default web browser at your localhost

### Step-by-Step Guide

#### 1. **Configure API Key**
   - Open the sidebar (click the `>` icon)
   - Enter your Gemini API key
   - Click "💾 Save API Key"

#### 2. **Add Documents**
   - Enter a blog URL in the "Add Documents" section
   - Click "➕ Add"
   - Wait for the documents to be processed and indexed
   - You'll see a success message when complete

#### 3. **Ask Questions**
   - Type your question in the "Ask Questions" section
   - Click "🚀 Search"
   - Wait for the AI to process and retrieve relevant information
   - View the answer displayed in a styled box

### Example Workflow

```python
# 1. Add a blog post
URL: https://example.com/blog-post

# 2. Ask questions
Question: "What are the main concepts discussed in this blog?"
Answer: [AI-generated response based on the content]

Question: "Can you summarize the key takeaways?"
Answer: [Contextual summary from the indexed documents]
```

---

## 🔧 How It Works

### Document Processing Pipeline

1. **Loading**: WebBaseLoader fetches content from the provided URL
2. **Splitting**: RecursiveCharacterTextSplitter divides content into chunks (100 tokens with 50 token overlap)
3. **Embedding**: Google Gemini embeddings convert text chunks into vectors
4. **Storage**: Vectors are stored in ChromaDB with unique IDs

### Query Processing Pipeline

1. **Query Input**: User submits a natural language question
2. **Agent Decision**: LangGraph agent decides if retrieval is needed
3. **Retrieval**: Vector similarity search finds relevant document chunks
4. **Relevance Check**: LLM evaluates if retrieved documents are relevant
5. **Query Rewrite** (if needed): If documents aren't relevant, query is improved
6. **Generation**: Final answer is generated using relevant context

### Key Technologies

- **LangChain**: Framework for building LLM applications
- **LangGraph**: State machine for orchestrating agent workflows
- **ChromaDB**: Open-source vector database for embeddings
- **Google Gemini**: LLM for embeddings and chat completions
- **Streamlit**: Web framework for the user interface

---

## 📁 Project Structure

```
rag/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── chroma_db/            # ChromaDB storage (created automatically)
│   └── ...               # Vector database files
│
└── .gitignore           # Git ignore file (recommended)
```

### Key Files

- **`app.py`**: Contains all application logic, UI components, and LangGraph workflow
- **`requirements.txt`**: Lists all Python package dependencies
- **`chroma_db/`**: Directory where ChromaDB stores vector embeddings (auto-created)

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError`
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

#### Issue: `API Key Error`
**Solution**: 
- Verify your Gemini API key is correct
- Check if the API key has proper permissions
- Ensure you've saved the key in the sidebar

#### Issue: `ChromaDB Initialization Error`
**Solution**:
```bash
# Clear the chroma_db directory and restart
rm -rf chroma_db/
# Then restart the app
```

#### Issue: `Document Loading Fails`
**Solution**:
- Verify the URL is accessible
- Check if the website blocks web scrapers
- Try a different URL format

#### Issue: `No Results Found`
**Solution**:
- Ensure documents have been added successfully
- Try rephrasing your question
- Check if the content is relevant to your query

### Performance Tips

- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` in `add_documents_to_vectorstore()` for different document types
- **Retrieval Count**: Modify `search_kwargs={"k": 5}` to retrieve more/fewer documents
- **Model Selection**: Change `model="gemini-2.0-flash"` to other Gemini models if needed

---

## 🛠️ Customization

### Adjusting Chunk Size

In `app.py`, modify the `RecursiveCharacterTextSplitter`:

```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200,  # Increase for longer chunks
    chunk_overlap=50  # Adjust overlap
)
```

### Changing Retrieval Parameters

Modify the retriever configuration:

```python
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # Retrieve more documents
)
```

### Customizing Prompts

Edit the prompt templates in:
- `grade_documents()`: Relevance checking prompt
- `rewrite()`: Query rewriting prompt
- `generate()`: Answer generation prompt

---

## 📊 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.3.0 | LLM framework |
| `langchain-google-genai` | ≥1.0.0 | Google Gemini integration |
| `langchain-chroma` | ≥0.1.0 | ChromaDB vector store |
| `langgraph` | ≥0.2.0 | Agent workflow orchestration |
| `streamlit` | ≥1.28.0 | Web UI framework |
| `chromadb` | ≥0.4.0 | Vector database |
| `pydantic` | ≥2.0.0 | Data validation |

See `requirements.txt` for complete list.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run the app in development mode
streamlit run app.py --server.runOnSave true
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **LangChain** team for the amazing framework
- **LangGraph** for agentic workflow capabilities
- **ChromaDB** for free vector storage
- **Google Gemini** for powerful LLM capabilities
- **Streamlit** for the beautiful UI framework

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [LangChain Documentation](https://python.langchain.com/)
3. Check [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
4. Open an issue on GitHub

---

## 🚀 Future Enhancements

Potential features for future versions:

- [ ] Support for multiple document formats (PDF, DOCX, etc.)
- [ ] Batch document upload
- [ ] Export conversation history
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] Document metadata search
- [ ] API endpoint for programmatic access
- [ ] User authentication
- [ ] Multiple vector stores support

---

<div align="center">

**Made with ❤️ using LangChain, LangGraph, and Streamlit**

⭐ Star this repo if you find it helpful!

</div>
