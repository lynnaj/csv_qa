import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

load_dotenv()

st.title("☁💸 Cloud Cost Chatbot POC")
st.markdown("#### Azure & AWS Cloud Cost Analysis with GPT-4o and LangGraph 🤖")

# st write magic
"""
I am using Streamlit UI and the official [`StreamlitCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_community.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html) 
within [_LangGraph_](https://langchain-ai.github.io/langgraph/) by leveraging callbacks in our 
graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).

---
"""

if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="Ask me anything about AWS or Azure cloud costs!")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handles graph events using the callback mechanism
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()  # Placeholder for visually updating AI's response after events end
        # create a new placeholder for streaming messages and other events, and give it context
        st_callback = get_streamlit_cb(st.empty())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))  # Add that last message to the st_message_state
        msg_placeholder.write(last_msg) # visually refresh the complete response after the callback container
