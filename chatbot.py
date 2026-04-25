import os 
import json 
import time 
import streamlit as st 

from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

# Load Api Through dotenv

load_dotenv()
Groq_Api_Key = os.getenv('GROQ_API_KEY','').strip() 

# Streamlit Page Config

st.set_page_config(
    page_title='Groq ChatBot AI',
    page_icon='https://img.icons8.com/?size=100&id=WmDgmNrDhz7f&format=png&color=000000',
    layout='centered'
)

st.title('Conversional AI Chatot')
st.caption('Built with Streamlit + Langchain + Groq Cloud API')

# Sidebar Control

st.sidebar.header('⚙️ Control')

New_Api = st.sidebar.text_input(
    'Groq API key ',
    type='password',
)

## [Value_if_True] if [Condition] else [Value_if_False]

Api_Input = New_Api if New_Api else Groq_Api_Key

Model = st.sidebar.selectbox(
    "Choose Model",
    [
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
    ]
)

temprature = st.sidebar.slider(
    "Temprature (Creativity)",
    max_value= 1.0,
    min_value= 0.0,
    value=0.0,
    step=0.05,
)

Max_Token = st.sidebar.slider(
    "Max Token (Reply Lenght)",
    max_value=1000,
    min_value=50,
    value=300,
    step=50,
)



system_prompt = st.sidebar.text_area(
    "Bot Rules",
    "Very Helpfull Ai Assistant Be Real Correct Clear And Concise",
)

Typing_Effect = st.sidebar.checkbox('Enable Typing Effect',value=True)

st.sidebar.divider()

if st.sidebar.button('Clear Chat'):
    st.session_state.pop('history_store',None)
    st.session_state.pop('download_cache',None)
    st.rerun()


if not Api_Input:
    st.error("❌ Groq API key is missing!")
    st.stop()

# Messge History 

if 'history_store' not in st.session_state:
    st.session_state.history_store = {}  

Session_ID = 'default_session'

def get_history(session_id):
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]


llm = ChatGroq(
    model=Model,
    api_key=Api_Input,
    temperature=temprature,
    max_tokens=Max_Token,
)

prompt = ChatPromptTemplate.from_messages([
    ('system',"{system_prompt}"),
    MessagesPlaceholder(variable_name='history'),
    ('human','{input}'),
])


chain = prompt | llm | StrOutputParser()


# Pipeline
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key='input',
    history_messages_key='history',
)

# Render Old Messge

history = get_history(Session_ID)

for message in history.messages:
    role = getattr(message,"type","")
    if role == "human":
        st.chat_message("user").write(message.content)

    else:
        st.chat_message("assistant").write(message.content)


# User Input

user_input = st.chat_input('Type Your Messege.... ')

if user_input:
    st.chat_message('user').write(user_input)

    with st.chat_message('assistant'):
        placeholder = st.empty()

        try:
        
            response = chat_with_history.invoke(
            {'input' : user_input, 'system_prompt' : system_prompt},
            config={'configurable' : {'session_id' : Session_ID}}
            )

        except Exception as e:
            st.error(str(e))
            response = ''

        if Typing_Effect and response:
            typed = ''

            for ch in response:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)
        
# Download Chat as json

st.divider()

st.subheader('Download Chat History')

export_data = []

for m in get_history(Session_ID).messages:
    role = getattr(m,'type',"")
    if role == 'human':
        export_data.append({'role' : 'user','text' : m.content})
    else:
        export_data.append({'role' : 'assistant', 'text' : m.content})

json_data = json.dumps(export_data,ensure_ascii= False, indent=2)

st.download_button(
    label='download chat history.json',
    data=json_data,
    file_name='history.json',
    mime='application/json',
)