# import streamlit as st
# import requests

# # Streamlit App Configuration
# st.set_page_config(page_title="LangGraph Agent UI", layout="centered")

# # Define API endpoint
# API_URL = "http://127.0.0.1:8000/chat"



# # Streamlit UI Elements
# st.title("LangGraph Chatbot Agent")
# st.write("Interact with the LangGraph-based agent using this interface.")

# # Input box for system prompt
# given_system_prompt = st.text_area("Define you AI Agent:", height=70, placeholder="Type your system prompt here...")

# # Predefined models
# MODEL_NAMES = [
#     "llama3-70b-8192",
#     "gemma2-9b-it"
# ]
# # Dropdown for selecting the model
# selected_model = st.selectbox("Select Model:", MODEL_NAMES)

# # Input box for user messages
# user_input = st.text_area("Enter your message(s):", height=150, placeholder="Type your message here...")

# # Button to send the query
# if st.button("Submit"):
#     if user_input.strip():
#         try:
#             # Send the input to the FastAPI backend
#             payload = {"messages": [user_input], "model_name": selected_model, 'system_prompt': given_system_prompt}
#             response = requests.post(API_URL, json=payload)

#             # Display the response
#             if response.status_code == 200:
#                 response_data = response.json()
#                 if "error" in response_data:
#                     st.error(response_data["error"])
#                 else:
#                     ai_responses = [
#                         message.get("content", "")
#                         for message in response_data.get("messages", [])
#                         if message.get("type") == "ai"
#                     ]

#                     if ai_responses:
#                         st.subheader("Agent Response:")
#                         st.markdown(f"**Final Response:** {ai_responses[-1]}")
#                         # for i, response_text in enumerate(ai_responses, 1):
#                         #     st.markdown(f"**Response {i}:** {response_text}")
#                     else:
#                         st.warning("No AI response found in the agent output.")
#             else:
#                 st.error(f"Request failed with status code {response.status_code}.")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#     else:
#         st.warning("Please enter a message before clicking 'Send Query'.")

import streamlit as st
import requests
import uuid
from datetime import datetime
import time

# Page config
st.set_page_config(page_title="💬 LangGraph Pro Chat", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #111;
}

.chat-container {
    background-color: #181818;
    padding: 1.2rem;
    border-radius: 10px;
    color: #f0f0f0;
    font-size: 1rem;
    max-height: 70vh;
    overflow-y: auto;
    box-shadow: 0 0 5px rgba(0,0,0,0.2);
}
.message {
    background-color: #222;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
    transition: 0.3s ease;
}
.message:hover {
    background-color: #2a2a2a;
}
.message strong {
    color: #bbb;
}
.message-time {
    font-size: 0.85rem;
    color: #777;
    margin-left: 10px;
}
.system-message {
    background-color: #282828;
    padding: 1rem;
    border-left: 5px solid #6c63ff;
    margin-bottom: 1rem;
    color: #ccc;
    font-family: monospace;
    font-style: italic;
}
input, textarea {
    color: #000;
}
.success-message, .error-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
}
.success-message {
    background-color: #198754;
    color: white;
}
.error-message {
    background-color: #dc3545;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Backend
API_URL = "http://127.0.0.1:8000"

# Session Init
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_info" not in st.session_state:
    st.session_state.model_info = {}
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# API helpers
def get_model_info():
    try:
        r = requests.get(f"{API_URL}/models")
        if r.status_code == 200:
            return r.json()["models"]
        return {}
    except:
        return {}

def fetch_chat_history():
    try:
        r = requests.get(f"{API_URL}/chat/history/{st.session_state.session_id}")
        if r.status_code == 200:
            data = r.json()
            st.session_state.chat_history = data["messages"]
            st.session_state.message_count = data["total_messages"]
        return True
    except:
        return False

def clear_chat_history():
    try:
        r = requests.delete(f"{API_URL}/chat/history/{st.session_state.session_id}")
        if r.status_code == 200:
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            return True
        return False
    except:
        return False

def send_message(message, model_name, system_prompt):
    try:
        payload = {
            "message": message,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "session_id": st.session_state.session_id,
            "max_history": 10
        }
        r = requests.post(f"{API_URL}/chat", json=payload)
        return r
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

# Header

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"Messages: {st.session_state.message_count}")
    
    if not st.session_state.model_info:
        st.session_state.model_info = get_model_info()
    
    model_names = list(st.session_state.model_info.keys()) if st.session_state.model_info else ["llama3-70b"]
    selected_model = st.selectbox("Model", model_names)
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant with deep knowledge and a friendly tone.",
        height=100
    )

    with st.expander("🧹 Session Tools"):
        if st.button("🔄 Refresh"):
            fetch_chat_history()
            st.success("History Refreshed!")
        if st.button("🗑️ Clear"):
            if clear_chat_history():
                st.success("History Cleared")
                st.rerun()
        if st.button("🔁 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.success("New Session Started")
            st.rerun()

# Input Area
st.markdown("### ✍️ Message")

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0
if "quick_action_message" not in st.session_state:
    st.session_state.quick_action_message = ""

user_input = st.text_area(
    "Your message...",
    key=f"input_{st.session_state.input_counter}",
    value=st.session_state.quick_action_message,
    height=100
)

send_button = st.button("🚀 Send", use_container_width=True)

# Chat Area
fetch_chat_history()
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# System message at top
if st.session_state.chat_history and st.session_state.chat_history[0]["type"] == "system":
    sys_msg = st.session_state.chat_history[0]
    sys_time = datetime.fromisoformat(sys_msg["timestamp"].replace("Z", "+00:00")).strftime("%H:%M:%S")
    st.markdown(f"""
        <div class="system-message">
            System • {sys_time}<br>{sys_msg["content"]}
        </div>
    """, unsafe_allow_html=True)

# All messages
for msg in reversed(st.session_state.chat_history):
    if msg["type"] == "system":
        continue
    time_str = ""
    if "timestamp" in msg:
        try:
            dt = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            time_str = dt.strftime('%H:%M:%S')
        except:
            pass
    speaker = "You" if msg["type"] == "human" else "AI"
    st.markdown(f"""
        <div class="message">
            <strong>{speaker}<span class="message-time"> • {time_str}</span></strong><br>
            {msg["content"]}
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Send Handler
if send_button and user_input.strip():
    with st.spinner("💬 AI is thinking..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)
        response = send_message(user_input, selected_model, system_prompt)
        progress.empty()

    if response and response.status_code == 200:
        result = response.json()
        st.markdown(f"""<div class="success-message">✅ Model: {result.get("model_used", selected_model)}</div>""", unsafe_allow_html=True)
        st.session_state.input_counter += 1
        st.session_state.quick_action_message = ""
        st.rerun()
    elif response:
        st.markdown(f"""<div class="error-message">❌ {response.status_code}: {response.text}</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="error-message">❌ Backend unreachable.</div>""", unsafe_allow_html=True)
elif send_button:
    st.warning("⚠️ Message is empty.")

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center; color:#888;">© 2025 LangGraph • Built for Production</div>', unsafe_allow_html=True)

