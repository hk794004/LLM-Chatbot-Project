import streamlit as st
import os
import json
import time
import dotenv

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()

Groq_Api_Key = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title='Groq Ai Chatbot',layout='wide')
st.title('🤖 Conversional Groq Chatbot')
st.caption('Built with Streamlit + Langchain + Groq Cloud API')

st.divider()

with st.sidebar:
    st.subheader('⚙️ Control')

    Api_Input = st.text_input(
        'Groq API KEY',
        type='password'
    )
        
Api_Key = Api_Input if Api_Input else Groq_Api_Key

        
if not Api_Key:
    st.warning('API Key Missing')
    st.stop()
else:
    st.sidebar.info('API Key Load')
 
with st.sidebar:

    Model = st.selectbox(
        'Choose Model',
        [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
        ]
        )

    Temprature = st.slider(
        'Temprature (Creativity)',
        max_value=0.99,
        min_value=0.0,
        value=0.0,
        step=0.01,
    )

    Token = st.slider(
        'Max Token (Reply lenght)',
        max_value=1024,
        min_value=64,
        value=64,
        step=1,
    )

    Rules = st.text_area(
        'Bot Rules',
        'You Are Very Helpfull Assistant please be clear and Quickly Ansawer',
    )

    Typing_Effect = st.checkbox('Enabled Typing Effect',value=True)

    col1,col2 = st.columns(2)

    with col1:
        Clear_Chat = st.button('Clear Chat')

User_Input = st.chat_input('Type Your Messege...')

LLM = ChatGroq(
    model=Model,
    api_key=Api_Key,
    temperature=Temprature,
    max_tokens=Token,
)

Session_ID = 'default'

if 'history_store' not in st.session_state:
    st.session_state['history_store'] = {}

history_store = st.session_state['history_store']

def get_history(session_id):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]


Prompt = ChatPromptTemplate.from_messages([
    ("system", Rules),
    MessagesPlaceholder(variable_name='history'),
    ("human","{input}"),
])

chain = Prompt | LLM | StrOutputParser()

Chat_History = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

History = get_history(Session_ID)

for message in History.messages:
    role = getattr(message, "type", "")

    if role=="human":
        st.chat_message("human").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)


if User_Input:
    st.chat_message('human').write(User_Input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:

            Response = Chat_History.invoke(
                {"input" : User_Input, "system" : Rules},
                config={"configurable" : {"session_id" : Session_ID}}
            )

        except Exception as e:
            st.error(str(e))
            Response = ""

        if Typing_Effect and Response:
            typed = ''

            for ch in Response:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)

export_data = []

for message in get_history(Session_ID).messages:
    role = getattr(message,"type","")

    if role=="human":
        export_data.append({"role" : "human", "text" : message.content})
    else:
        export_data.append({"role" : "assistant", "text" : message.content})

json = json.dumps(export_data,ensure_ascii=False,indent=2)

with col2:
    
    st.download_button(
    label='Chat History',   
    data=json,
    file_name='chat_history.json'
    )

if Clear_Chat:
    st.session_state.pop('history_store',None)
    st.rerun()



    

