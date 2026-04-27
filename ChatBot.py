import os
import json
import time
import dotenv
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

# Load GROQ_API

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Streamlit Page Setup

st.set_page_config("GROQ AI ChatBot",layout="centered")
st.title("🤖 Conversional Groq AI ChatBot")
st.caption("Build With Streamlit + Langchain + GROQ API Cloud")

st.divider()

# Build Sidebar

with st.sidebar:

    st.subheader("⚙️ Control")

    api_input = st.text_input(
        "GROQ API KEY",
        type="password",
    )

    API_Key = api_input if api_input else GROQ_API_KEY

if not API_Key:
    st.warning("API Key Missing Insert into The Sidebar")
    st.stop()
else:
    st.sidebar.success("API Key Loaded")

with st.sidebar:

    Models = st.selectbox(
        "Choose Models",
        ["meta-llama/llama-4-scout-17b-16e-instruct","qwen/qwen3-32b"],
    )

    Temprature = st.slider(
        "Temprature (Creativity)",
        max_value=0.99,
        min_value=0.0,
        value=0.0,
        step=0.01,
    )

    Max_Tokens = st.slider(
        "Max Tokens (Reply Length)",
        max_value=2000,
        min_value=64,
        value=500,
        step=1,
    )

    Bot_Rules = st.text_area(
        "Bot Rules",
        "Users must communicate in a polite and respectful manner at all times. Any form of harassment, hate speech, abusive language, or offensive remarks is strictly prohibited.",
    )

    Bot_Tone = st.selectbox(
        "Bot Tone",
        [
            "Friendly",
            "Strict",
            "Professional",
        ]
    )

    Typing_Effect = st.checkbox("Enable Typing Effect",value=True)

    st.divider()

    cola,colb = st.columns(2)

    with cola:

        Clear_Chat = st.button("Clear Chat")

User_Input = st.chat_input("Type Your Message Here...")

# Call LLM Model

LLM = ChatGroq(
    model=Models,
    api_key=API_Key,
    temperature=Temprature,
    max_tokens=Max_Tokens,
)

if "history" not in st.session_state:
    st.session_state["history"] = {}

history = st.session_state["history"]

def get_history(session_id):
    if session_id not in history:
        history[session_id] = InMemoryChatMessageHistory()
    return history[session_id]

History = get_history("default")

for message in History.messages:
    role = getattr(message,"type","")

    if role == "human":
        st.chat_message("human").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)

Tones = {

    "Friendly" : "Please be kind and respectful when interacting. Positive and polite conversations help create a better experience for everyone. The bot is here to help and appreciates friendly communication",
    "Strict" : "All users must maintain respectful communication. Any use of abusive language, harassment, or inappropriate behavior will not be tolerated. Violations may result in restricted or terminated access",
    "Professional" : "Users are expected to communicate in a clear, respectful, and professional manner at all times. Any form of abusive, inappropriate, or unprofessional language is not permitted. Failure to follow these guidelines may result in restricted access to the bot’s services.",

}

Behaviour = Bot_Rules + Tones[Bot_Tone]

Prompt = ChatPromptTemplate.from_messages([
    ("system", Behaviour),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}"),
])

Chain = Prompt | LLM | StrOutputParser()

Chat_History = RunnableWithMessageHistory(
    Chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)


# Streamlit UI Input

if User_Input:
    st.chat_message("human").write(User_Input)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        response = Chat_History.invoke(
            {"input" : User_Input, "system" : Behaviour},
            config={"configurable" : {"session_id" : "default"}}
        )

        if Typing_Effect and response:
            type = ""

            for ch in response:
                type += ch
                placeholder.markdown(type)
                time.sleep(0.005)
        else:
            st.write(response)

if Clear_Chat:
    st.session_state.pop("history",None)

# Download Chat History

export_data = []

for message in get_history('default').messages:
    role = getattr(message,"type","")

    if role == "human":
        export_data.append({"role" : "human", "text" : message.content})
    else:
        export_data.append({"role" : "assistant", "text" : message.content})


Json = json.dumps(export_data)

with st.sidebar:

    with colb:
        
        Download_Button = st.download_button(
            "chat history",
            data=Json,
            file_name="chat_history.txt"
        )

if Clear_Chat:
    st.session_state.pop("history",None)
    st.session_state.pop("download_cache",None)
    st.rerun()