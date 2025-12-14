"""
AI Assistant Page – Streamlit
Course: CST1510
Week 10: AI Integration

Features:
- Secure OpenAI API usage via Streamlit secrets
- Chat-style AI assistant
- Conversation memory
- AI-assisted database updates (with user confirmation)
"""

import streamlit as st
import sqlite3
import json
from openai import OpenAI



# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Assistant")
st.caption("Multi-Domain Intelligence Platform")



# OPENAI CLIENT (NEW API – CORRECT)
client = OpenAI(
    api_key=st.secrets["OPEN_AI_KEY"]
)


# DATABASE CONFIG
DB_PATH = "DATA/intelligence_platform.db"


def update_table(table_name: str, data: dict):
    """
    Generic UPDATE helper.
    Assumes the table has an 'id' column.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    fields = ", ".join([f"{k}=?" for k in data if k != "id"])
    values = [v for k, v in data.items() if k != "id"]
    values.append(data["id"])

    query = f"UPDATE {table_name} SET {fields} WHERE id=?"
    cursor.execute(query, values)

    conn.commit()
    conn.close()



# SESSION STATE (CHAT MEMORY)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for a university Multi-Domain "
                "Intelligence Platform.\n\n"
                "You help with:\n"
                "- Cybersecurity incident analysis\n"
                "- Dataset metadata explanations\n"
                "- IT ticket troubleshooting\n\n"
                "IMPORTANT:\n"
                "- If asked to update data, respond ONLY with valid JSON.\n"
                "- JSON must include an 'id' field.\n"
                "- Example:\n"
                "{ \"id\": 2, \"status\": \"Resolved\" }"
            )
        }
    ]



# SIDEBAR CONTROLS
with st.sidebar:
    st.header("💬 Chat Controls")

    msg_count = len(
        [m for m in st.session_state.messages if m["role"] != "system"]
    )
    st.metric("Messages", msg_count)

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

    st.divider()
    st.header("🧠 AI Table Updates")

    enable_updates = st.checkbox("Enable AI-assisted updates")
    target_table = st.selectbox(
        "Select table",
        ["None", "cyber_incidents", "datasets_metadata", "it_tickets"]
    )



# DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# USER INPUT
user_input = st.chat_input(
    "Ask me about cyber incidents, datasets, or IT tickets..."
)

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    
    # OPENAI RESPONSE (NEW RESPONSES API)
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=st.session_state.messages
            )

            ai_reply = response.output_text
            st.markdown(ai_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": ai_reply}
    )

    
    # APPLY AI UPDATE
    if enable_updates and target_table != "None":
        try:
            update_data = json.loads(ai_reply)

            if "id" in update_data:
                st.warning("⚠ AI suggests a database update")
                st.json(update_data)

                if st.button("✅ Confirm Update"):
                    update_table(target_table, update_data)
                    st.success(
                        f"Table '{target_table}' updated successfully."
                    )

        except json.JSONDecodeError:
            pass  # Normal chat response
