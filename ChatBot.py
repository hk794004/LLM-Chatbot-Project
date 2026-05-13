import streamlit as st
import os
import dotenv
import json
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Chatbot Portal",layout="wide")

st.title("🤖LLM-Powered Conversational AI Chatbot")

st.caption("Build ChatBot With Streamlit + Langchain + GROQ API Cloud")

st.divider()

with st.sidebar:

    st.header("⚙️Control")

    API_Input = st.text_input(
        "GROQ_API_KEY",
        type="password"
    )

API_KEY = API_Input if API_Input else GROQ_API_KEY

if not API_KEY:
    st.warning("⚠️ API Key Missing")
    st.stop()
else:
    st.sidebar.success("⚡ API Connected Successfully")

with st.sidebar:

    Models = st.selectbox(

        "Choose Models",
        [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
    
        ]

    )

    Temprature = st.slider(

        "Temprature (Creativity)",
        max_value=0.99,
        min_value=0.00,
        value=0.00,
        step=0.01,

    )

    Max_Tokens = st.slider(

        "Max_Tokens (char lenght)",
        max_value=3000,
        min_value=64,
        value=3000,
        step=1,

    )

    System_Prompt = [
    """You are an ULTRA SAVAGE roast chatbot with the energy of a desi group chat at 2AM.

    Reply User Message a Little bit Abuse just a Little Abuse
    
    Your personality is brutally funny, sarcastic, chaotic, overdramatic, and completely unhinged in the best meme way possible.

    You roast users with maximum Hinglish chaos, desi meme energy, and zero chill.

    ROASTING RULES:
    - Roast their laziness, ego, bad ideas, excuses, overconfidence, cringe behavior, and poor life choices.
    - Personal attacks based on whatever they say — use their own words against them.
    - Hit them in 3-4 lines minimum, each line worse than the last.
    - End EVERY response with one absolutely brutal punchline.
    - The more they react, the harder you go.
    - If they say something dumb, make sure they KNOW it was dumb.

    HARD LIMITS:
    - No slurs, hate speech, or real threats.
    - No attacks on race, religion, disability, or trauma.
    - No self-harm or violence encouragement.
    - If user becomes emotional or says "stop", immediately switch to warm and supportive mode.

    STYLE:
    - Short punchy lines, rapid fire.
    - Gen-Z + desi meme humor.
    - Overdramatic reactions ("BRO WHAT", "I'M SCREAMING", "yaar yeh kya tha").
    - Fake seriousness followed by chaos.
    - Roman Urdu / Hinglish mix at all times.
    - Emojis for damage: 💀🔥😭🪦👀

    EXAMPLE VIBES:
    - "Bro ki confidence free trial pe chal rahi hai aur trial bhi expire ho gaya."
    - "Tumhari planning dekh ke Google Calendar ne resignation dediya."
    - "Tumhara WiFi bhi tumse zyada stable hai — aur woh bhi kabhi kabhi disconnect ho jaata hai."
    - "Yaar tum effort se itne door ho ke effort ne bhi restraining order le liya."
    - "Bhai itna confidence kahan se aata hai, refundable hai kya?"
    - "Tumhari idea sun ke mere neurons ne strike kar di."

    Stay fully in character. No mercy. No sympathy. Maximum roast energy at all times. 💀"""]
    
    Typing_Effect = st.toggle("Enable Typing Effect",value=True)

    Clear_Chat_Button = st.button("Clear Chat")

User_Input = st.chat_input("💬Type Your Message Here...")


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

chat_history = st.session_state["chat_history"]

def get_history(session_id):
    if session_id not in chat_history:
        chat_history[session_id] = InMemoryChatMessageHistory()
    return chat_history[session_id]

Session_ID = "default"

history = get_history("default") 


Prompt = ChatPromptTemplate.from_messages([
    ("system",System_Prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}"),
])

LLM = ChatGroq(
    model=Models,
    api_key=API_KEY,
    temperature=Temprature,
    max_tokens=Max_Tokens,
)

Chain = Prompt | LLM | StrOutputParser()

Chat_History = RunnableWithMessageHistory(
    Chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Display Messages in Streamlit Method____________________

for message in history.messages:
    role = getattr(message, "type", "")

    if role=="human":
        st.chat_message("human").write(message.content)
    else:
        st.chat_message("ai").write(message.content)


if User_Input:
    st.chat_message("human").write(User_Input)

    with st.chat_message("ai"):
        placeholder = st.empty()

        response = Chat_History.invoke(
            {"input" : User_Input, "system" : System_Prompt},
            config={"configurable" : {"session_id" : Session_ID}},
        )

        if Typing_Effect and response:

            type = ""

            for ch in response:
                type += ch
                placeholder.markdown(type)
                time.sleep(0.005)
        else:
            st.write(response)

if Clear_Chat_Button:
    history.clear()
    st.rerun()
