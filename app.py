# # Import necessary modules and setup for FastAPI, LangGraph, and LangChain
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import List, Dict, Any, Optional
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# import os
# import json
# from datetime import datetime
# from langgraph.prebuilt import create_react_agent
# from langchain_groq import ChatGroq
# import uvicorn
# from dotenv import load_dotenv
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# # Retrieve and set API keys for external tools and services
# groq_api_key = os.getenv("GROQ_API_KEY")
# tavily_api_key = os.getenv("TAVILY_API_KEY")

# if not groq_api_key:
#     raise ValueError("GROQ_API_KEY environment variable is required")
# if not tavily_api_key:
#     raise ValueError("TAVILY_API_KEY environment variable is required")

# os.environ["TAVILY_API_KEY"] = tavily_api_key

# # Enhanced model configuration with descriptions
# MODEL_CONFIGS = {
#     "llama3-70b-8192": {
#         "name": "Llama 3 70B",
#         "description": "Large language model with 70B parameters, excellent for complex reasoning",
#         "max_tokens": 8192
#     },
#     "mixtral-8x7b-32768": {
#         "name": "Mixtral 8x7B",
#         "description": "Mixture of experts model with high context length",
#         "max_tokens": 32768
#     },
#     "gemma2-9b-it": {
#         "name": "Gemma 2 9B IT",
#         "description": "Instruction-tuned model optimized for conversations",
#         "max_tokens": 8192
#     }
# }

# # Initialize tools
# tool_tavily = TavilySearchResults(max_results=3)
# tools = [tool_tavily]

# # In-memory storage for chat sessions (in production, use a database)
# chat_sessions: Dict[str, List[Dict]] = {}

# # FastAPI application setup
# app = FastAPI(
#     title='Enhanced LangGraph AI Agent',
#     description='Advanced AI agent with context awareness and chat history',
#     version='2.0.0'
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Enhanced request schemas
# class MessageRequest(BaseModel):
#     content: str = Field(..., description="The message content")
#     type: str = Field(default="human", description="Message type: human, ai, or system")

# class ChatRequest(BaseModel):
#     system_prompt: str = Field(..., description="System prompt for the AI agent")
#     model_name: str = Field(..., description="Name of the model to use")
#     message: str = Field(..., description="Current user message")
#     session_id: str = Field(..., description="Unique session identifier")
#     max_history: int = Field(default=10, description="Maximum number of previous messages to include")

# class ChatResponse(BaseModel):
#     response: str
#     session_id: str
#     model_used: str
#     timestamp: str
#     message_count: int

# class SessionHistoryResponse(BaseModel):
#     session_id: str
#     messages: List[Dict[str, Any]]
#     total_messages: int

# class ModelInfo(BaseModel):
#     models: Dict[str, Dict[str, Any]]

# def format_chat_history(session_id: str, max_history: int = 10) -> List:
#     """Format chat history for the LangGraph agent"""
#     if session_id not in chat_sessions:
#         return []
    
#     # Get the last max_history messages
#     recent_messages = chat_sessions[session_id][-max_history:]
    
#     formatted_messages = []
#     for msg in recent_messages:
#         if msg["type"] == "human":
#             formatted_messages.append(HumanMessage(content=msg["content"]))
#         elif msg["type"] == "ai":
#             formatted_messages.append(AIMessage(content=msg["content"]))
#         elif msg["type"] == "system":
#             formatted_messages.append(SystemMessage(content=msg["content"]))
    
#     return formatted_messages

# def save_message_to_session(session_id: str, content: str, message_type: str):
#     """Save a message to the chat session"""
#     if session_id not in chat_sessions:
#         chat_sessions[session_id] = []
    
#     message = {
#         "content": content,
#         "type": message_type,
#         "timestamp": datetime.now().isoformat()
#     }
    
#     chat_sessions[session_id].append(message)
#     logger.info(f"Saved {message_type} message to session {session_id}")

# @app.get("/models", response_model=ModelInfo)
# def get_available_models():
#     """Get information about available models"""
#     return ModelInfo(models=MODEL_CONFIGS)

# @app.post("/chat", response_model=ChatResponse)
# def chat_endpoint(request: ChatRequest):
#     """
#     Enhanced API endpoint with context awareness and chat history
#     """
#     try:
#         # Validate model name
#         if request.model_name not in MODEL_CONFIGS:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid model name. Available models: {list(MODEL_CONFIGS.keys())}"
#             )

#         # Initialize the LLM with the selected model
#         llm = ChatGroq(
#             groq_api_key=groq_api_key, 
#             model=request.model_name,
#             temperature=0.7,
#             max_tokens=MODEL_CONFIGS[request.model_name]["max_tokens"]
#         )

#         # Get chat history
#         chat_history = format_chat_history(request.session_id, request.max_history)
        
#         # Add system message if this is the first message in the session
#         if not chat_history:
#             chat_history.append(SystemMessage(content=request.system_prompt))
#             save_message_to_session(request.session_id, request.system_prompt, "system")

#         # Add current user message
#         chat_history.append(HumanMessage(content=request.message))
#         save_message_to_session(request.session_id, request.message, "human")

#         # Create agent with system prompt
#         llm_with_prompt = llm.bind(system_message=request.system_prompt)
#         agent = create_react_agent(llm_with_prompt, tools=tools)

#         # Prepare state with chat history
#         state = {"messages": chat_history}

#         # Get response from agent
#         result = agent.invoke(state)
        
#         # Extract AI response
#         ai_response = ""
#         if "messages" in result:
#             for message in reversed(result["messages"]):
#                 if hasattr(message, 'type') and message.type == "ai":
#                     ai_response = message.content
#                     break
#                 elif isinstance(message, dict) and message.get("type") == "ai":
#                     ai_response = message.get("content", "")
#                     break

#         if not ai_response:
#             ai_response = "I apologize, but I couldn't generate a proper response. Please try again."

#         # Save AI response to session
#         save_message_to_session(request.session_id, ai_response, "ai")

#         return ChatResponse(
#             response=ai_response,
#             session_id=request.session_id,
#             model_used=request.model_name,
#             timestamp=datetime.now().isoformat(),
#             message_count=len(chat_sessions.get(request.session_id, []))
#         )

#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/chat/history/{session_id}", response_model=SessionHistoryResponse)
# def get_chat_history(session_id: str):
#     """Get chat history for a specific session"""
#     if session_id not in chat_sessions:
#         return SessionHistoryResponse(
#             session_id=session_id,
#             messages=[],
#             total_messages=0
#         )
    
#     return SessionHistoryResponse(
#         session_id=session_id,
#         messages=chat_sessions[session_id],
#         total_messages=len(chat_sessions[session_id])
#     )

# @app.delete("/chat/history/{session_id}")
# def clear_chat_history(session_id: str):
#     """Clear chat history for a specific session"""
#     if session_id in chat_sessions:
#         del chat_sessions[session_id]
#         return {"message": f"Chat history cleared for session {session_id}"}
#     else:
#         raise HTTPException(status_code=404, detail="Session not found")

# @app.get("/chat/sessions")
# def get_all_sessions():
#     """Get all active chat sessions"""
#     sessions = []
#     for session_id, messages in chat_sessions.items():
#         if messages:
#             last_message = messages[-1]
#             sessions.append({
#                 "session_id": session_id,
#                 "message_count": len(messages),
#                 "last_activity": last_message["timestamp"]
#             })
    
#     return {"sessions": sessions}

# @app.get("/health")
# def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "active_sessions": len(chat_sessions)
#     }

# if __name__ == '__main__':
#     uvicorn.run(
#         app, 
#         host='127.0.0.1', 
#         port=8000,
#         reload=True,
#         log_level="info"
#     )

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from datetime import datetime
import os, uuid, uvicorn, logging
from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safely load environment variables (always as strings)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is required")
tavily_key = os.getenv("TAVILY_API_KEY", "")
os.environ["TAVILY_API_KEY"] = tavily_key  # always string :contentReference[oaicite:1]{index=1}

# Models config
MODEL_CONFIGS = {
    "llama3-70b-8192": {"max_tokens": 8192},
    "mixtral-8x7b-32768": {"max_tokens": 32768},
    "gemma2-9b-it": {"max_tokens": 8192},
}

# Tools configuration
web_tool = DuckDuckGoSearchResults(max_results=3)
arxiv_tool = ArxivQueryRun()
tools = [web_tool, arxiv_tool]

# Session storage
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

# FastAPI app
app = FastAPI(title="Agentic AI with DuckDuckGo & ArXiv")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Request/Response models
class ChatRequest(BaseModel):
    system_prompt: str
    model_name: str
    message: str
    session_id: str
    max_history: int = 10

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    timestamp: str
    message_count: int

# Helper functions
def format_history(sid: str, max_hist: int):
    recent = chat_sessions.get(sid, [])[-max_hist:]
    formatted = []
    for m in recent:
        cls = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}[m["type"]]
        formatted.append(cls(content=m["content"]))
    return formatted

def save_msg(sid: str, content: str, typ: str):
    chat_sessions.setdefault(sid, []).append({
        "content": content, "type": typ, "timestamp": datetime.now().isoformat()
    })
    logger.info(f"Saved {typ} message in session {sid}")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if req.model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="Invalid model")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=req.model_name,
        temperature=0.7,
        max_tokens=MODEL_CONFIGS[req.model_name]["max_tokens"]
    )

    history = format_history(req.session_id, req.max_history)
    if not history:
        history.append(SystemMessage(content=req.system_prompt))
        save_msg(req.session_id, req.system_prompt, "system")

    history.append(HumanMessage(content=req.message))
    save_msg(req.session_id, req.message, "human")

    llm_bound = llm.bind(system_message=req.system_prompt)
    agent = create_react_agent(llm_bound, tools=tools)
    result = agent.invoke({"messages": history})

    ai_resp = ""
    if "messages" in result:
        for m in reversed(result["messages"]):
            if getattr(m, "type", None) == "ai":
                ai_resp = m.content
                break
    if not ai_resp:
        ai_resp = "Sorry, I couldn't generate a response."

    save_msg(req.session_id, ai_resp, "ai")

    return ChatResponse(
        response=ai_resp,
        session_id=req.session_id,
        model_used=req.model_name,
        timestamp=datetime.now().isoformat(),
        message_count=len(chat_sessions[req.session_id])
    )

# Models info
@app.get("/models")
def get_models():
    return {"models": MODEL_CONFIGS}

# Fetch history
@app.get("/chat/history/{session_id}")
def get_history(session_id: str):
    msgs = chat_sessions.get(session_id, [])
    return {"session_id": session_id, "messages": msgs, "total_messages": len(msgs)}

# Clear history
@app.delete("/chat/history/{session_id}")
def clear_history(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"History cleared for {session_id}"}
    raise HTTPException(status_code=404, detail="Session not found")

# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "active_sessions": len(chat_sessions)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

