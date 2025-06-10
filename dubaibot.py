import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

initial_message = [
    {"role": "system", "content": "you are trip planner in dubai,you know about hotels,tourist spot,guides etc.your name is dubai genie,also known as DG.you should respond within 200 words,always ask question to user and make them comfortable,deal professionally."},
    {"role": "assistant", "content": "hello im DG,your expert trip planner,how can i help you ."}
]

def get_response_from_llm(messages):
    completion = client.chat.completions.create(
        model="gpt-4",

        messages=messages
    )
    return completion.choices[0].message.content


if "messages" not in st.session_state:
    st.session_state.messages = initial_message

st.title("Dubai Trip Assistant")


for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_message = st.chat_input("Enter your message")
if user_message:
    new_message = {
        "role": "user",
        "content": user_message
    }
    st.session_state.messages.append(new_message)

    with st.chat_message("user"):
        st.markdown(user_message)

    response = get_response_from_llm(st.session_state.messages)
    if response:
        response_message = {
            "role": "assistant",
            "content": response
        }
        st.session_state.messages.append(response_message)

        with st.chat_message("assistant"):
            st.markdown(response)
