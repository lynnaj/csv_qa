import os
import streamlit as st

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import AzureChatOpenAI

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import interrupt

from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, Literal
from typing_extensions import TypedDict

import pandas as pd 
from PIL import Image
import uuid

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

df_aws = pd.read_csv("AWSUsage_outlier.csv")[['Service', 'Region', 'Cost', 'Date', 'Project', 'Week']]
df_azure = pd.read_csv("AzureUsage.csv")

df_template = r"""```python
str({df_name}.columns.tolist())
>>> {df_columns}
```"""

df_context = "\n\n".join(
    df_template.format(df_columns=str(_df.columns.tolist()), df_name=df_name)
    for _df, df_name in [(df_aws, "df_aws"), (df_azure, "df_azure")]
)

df_aws['Date'] = pd.to_datetime(df_aws['Date'])
df_azure['Date'] = pd.to_datetime(df_azure['Date'])

ls_aws_services = df_aws['Service'].unique().tolist()
ls_azure_services = df_azure['ServiceName'].unique().tolist()

max_aws_date = str(df_aws['Date'].max().strftime('%Y-%m-%d'))
max_azure_date = str(df_azure['Date'].max().strftime('%Y-%m-%d'))

filename = str(uuid.uuid4())

system = f"""You have access to two datasets about Cloud Spend.  They are already loaded as a pandas dataframe, use df_aws and df_azure. \
When users asks about budget data, please inform them that you don't have access to budget data. \
Do not create your own sample datasets. \
The last date of df_aws data goes until {max_aws_date} \
The last date of df_azure data goes until {max_aws_date} \
Any queries of data after those dates, you do not have. \
Do not forecast, and tell user that you do not have the data. \
Here is a list of columns for each dataframe and the python code that was used to generate the list:
{df_context} \
Here is a list of AWS services: {ls_aws_services} \
Here is a list of Azure services: {ls_azure_services} \
Given a user question about the data, write the Python code to answer it.  Prioritize output the results as a dataframe. When outputting strings, use print() python code.\
When outputting dataframes, do not use print(). \
Unless the user ask for a chart or graph, then use matplotlib for streamlit. \
Be sure to include import matplotlib.pyplot as plt and use plt.savefig("image/{filename}.png") \
Explicitly close the figure with plt.close() and print("image saved") \
Don't assume you have access to any libraries other than built-in Python. To use pandas, please import pandas as pd. When aggregating over time, be sure to pass numeric_only=True . \
For example, here is pandas code for aggregating spend by month over month: df_aws['Costs'].groupby(df_aws['Dates'].dt.to_period('M')).sum(numeric_only=True).reset_index() \
Dates column has been already converted to pandas datetime object. \
Be sure to convert date period object to timestamp before visualizing, like this, df_aws_redshift_monthly['Dates'] = df_aws_redshift_monthly['Dates'].dt.to_timestamp() \
The x-axis labels should be angled so the labels do not overlap. \
Only annotate long line and bar graphs with min and max, rounded to whole number. Use bbox = dict(boxstyle="round", fc="0.8") and declare the following within annonate(): arrowprops=dict(facecolor='black', shrink=0.05), bbox=bbox \
When cost is being asked, included dollar signs to the numbers and round to whole number. \
Create appropriate titles for the charts, and use plt.tight_layout() to ensure the layout is clean. \
Make sure to refer only to the variables mentioned above."""


repl_tool = PythonAstREPLTool(
    locals={"df_aws": df_aws, "df_azure": df_azure},
    name="python_repl",
    description="Runs Python code and returns the output of the final line."
)

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str
    

def get_image_data(image_file_name: str) -> bytes:
    """Loads image data from the specified image name in the 'image' folder and displays it in Streamlit."""
    image_path = os.path.join("image", image_file_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_file_name}' not found.")
    image = Image.open(image_path)
    st.image(image, caption='The graph was created with the assistance of AI tools.  Please verify information.')
    return f"Image successfully loaded: {image_file_name}"
    #with open(image_path, "rb") as f:
    #    return f.read()

# System message
sys_msg = SystemMessage(content=system)

# List of tools that will be accessible to the graph via the ToolNode
tools = [repl_tool, AskHuman, get_image_data]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Core invocation of the model
def assistant(state: GraphsState):
    messages = state["messages"]
    llm = AzureChatOpenAI(
        azure_deployment='gpt-4o',
        api_version='2024-12-01-preview',
        azure_endpoint='xxx',
        api_key='xxx',
        temperature=0,
        streaming=True,
    ).bind_tools(tools)
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "tools"

# We define a fake node to ask the human
def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
    location = interrupt(ask.question)
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("ask_human", ask_human)

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
#builder.add_conditional_edges(
#    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
#    tools_condition,
#)

builder.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "assistant",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    path_map=["ask_human", "tools", END],
)

builder.add_edge("tools", "assistant")

# After we get back the human response, we go back to the agent
builder.add_edge("ask_human", "assistant")

react_graph = builder.compile()

# Function to invoke the compiled graph externally
def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return react_graph.invoke({"messages": st_messages}, config={"callbacks": callables})
