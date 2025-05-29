import streamlit as st
from langchain.chat_models import init_chat_model
import pydantic
import time
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
import matplotlib.pyplot as plt
from langchain.callbacks import StreamlitCallbackHandler
import statsmodels.api as sm

model_id = "amazon.nova-pro-v1:0"

df_azure = pd.read_csv("AzureUsage.csv")
df_aws = pd.read_csv("AWSUsage.csv")[['Services', 'Regions', 'Costs', 'Dates']]

# Initialize BedrockChat model
llm = init_chat_model(
    "amazon.nova-pro-v1:0",
    model_provider="bedrock",
    model_kwargs={'temperature': 0.7}
)


tool = PythonAstREPLTool(locals={"df_aws": df_aws, "df_azure": df_azure}, verbose=True)

parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

llm_with_tool = llm.bind_tools(tools=[tool], tool_choice=tool.name)

df_template = """\`\`\`python
str({df_name}.columns.tolist())
>>> {df_columns}
\`\`\`"""

df_context = "\n\n".join(
    df_template.format(df_columns=str(_df.columns.tolist()), df_name=df_name)
    for _df, df_name in [(df_aws, "df_aws"), (df_azure, "df_azure")]
)

system = f"""You have access to two datasets about Cloud Spend.  They are loaded as pandas dataframes, df_azure and df_aws. \
Do not create your own sample datasets. \
Here is a list of columns for each dataframe and the python code that was used to generate the list:
{df_context}
Given a user question about the data, write the Python code to answer it.  Prioritize output the results as a dataframe. Avoid outputing strings.\
Unless the user ask for a chart or graph, then use matplotlib for streamlit. \
Be sure to include import matplotlib.pyplot as plt and end code with st.pyplot(fig)  \
Don't assume you have access to any libraries other than built-in Python. To use pandas, please import pandas as pd. To use datetime, please import datetime as dt. \
To forecast, please import statsmodels.api as sm. Note that disp is no longer support.  Use model.fit()\
Make sure to refer only to the variables mentioned above."""

#print("system prompt: " + system)

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
                                           #, MessagesPlaceholder(variable_name="chat_history")])

chain = prompt | llm_with_tool | parser | tool


st.title("Chat with a dataset")

# Initialize chat history
#if "messages" not in st.session_state:
#    st.session_state.messages = []

# Display chat messages from history on app rerun
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"], unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
# Accept user input
if query := st.chat_input("Ask me about any of the Cloud Spend Datasets"):
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query, unsafe_allow_html=True)
    # Add user message to chat history
    #st.session_state.messages.append({"role": "user", "content": query})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = chain.invoke({"question": query, }, {"callbacks":[st_cb]}) #"chat_history": st.session_state.messages})
        print("TYPE: " + str(type(response)))
        if isinstance(response, str):
            try:
                response = st.pyplot(response)
            except Exception as e:
                st.error(f"Error displaying plot: {e}")
                response = st.write(response)
        else:
            response = st.write(response)
    # Add assistant response to chat history
    #st.session_state.messages.append({"role": "assistant", "content": response})
