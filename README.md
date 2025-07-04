# LangGraph Agent

A production-ready conversational AI agent built with FastAPI, Streamlit, LangGraph, and LangChain, containerized with Docker for easy deployment.

## Overview

This project implements an intelligent AI agent that combines:
- **FastAPI backend**: Handles chat management, LLM integration, and web search capabilities
- **Streamlit UI**: Provides a responsive chat interface with session management
- **LangGraph + LangChain**: Powers the reasoning and tool usage capabilities
- **Docker container**: Simplifies deployment across environments

The agent can search the web via DuckDuckGo and access academic papers through ArXiv to provide up-to-date information.

## Features

- 🧠 LLM-powered agent with ReAct reasoning pattern
- 🔍 Web search via DuckDuckGo integration
- 📚 Academic paper search via ArXiv
- 💬 Persistent chat sessions with history management
- 🔄 Multiple LLM model support (Llama3, Mixtral, Gemma)
- 🐳 Docker containerization for easy deployment
- 🎨 Modern Streamlit UI with dark mode

## Project Structure

```
.
├── app.py            # FastAPI backend server
├── ui.py             # Streamlit UI implementation
├── Dockerfile        # Docker configuration
├── requirements.txt  # Python dependencies
├── .env              # Environment variables (API keys)
└── .gitignore        # Git ignore configuration
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- API keys:
  - Groq API key
  - (Optional) Tavily API key

### Running with Docker

1. Clone this repository
2. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
3. Build and run the Docker container:
   ```bash
   docker build -t langgraph-agent .
   docker run -p 8000:8000 -p 8501:8501 langgraph-agent
   ```
4. Access the UI at http://localhost:8501

### Running without Docker

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. In a separate terminal, run the UI:
   ```bash
   streamlit run ui.py
   ```

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /chat` - Send a message to the AI agent
- `GET /models` - Get information about available models
- `GET /chat/history/{session_id}` - Retrieve chat history for a session
- `DELETE /chat/history/{session_id}` - Clear chat history for a session
- `GET /health` - Health check endpoint

## Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web API framework
- [Streamlit](https://streamlit.io/) - UI framework for ML applications
- [LangChain](https://www.langchain.com/) - LLM application framework
- [LangGraph](https://python.langchain.com/docs/langgraph) - Multi-agent orchestration
- [Groq](https://groq.com/) - Fast LLM inference provider
- [Docker](https://www.docker.com/) - Containerization platform

## Future Improvements

- Add authentication and multi-user support
- Implement persistent storage with a database
- Enhance error handling and retries
- Add additional tools and capabilities

## License

This project is open source and available under the [MIT License](LICENSE).
