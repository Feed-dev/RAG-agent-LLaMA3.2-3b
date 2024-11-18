#!/usr/bin/env python
# coding: utf-8

# # LangChain: Agents
# 
# ## Outline:
# 
# * Using built in LangChain tools: DuckDuckGo search and Wikipedia
# * Defining your own tools

# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[ ]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


# ## Built-in LangChain tools

# In[ ]:


#!pip install -U wikipedia


# In[ ]:


from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI


# In[ ]:


llm = ChatOpenAI(temperature=0, model=llm_model)


# In[ ]:


tools = load_tools(["llm-math","wikipedia"], llm=llm)


# In[ ]:


agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)


# In[ ]:


agent("What is the 25% of 300?")


# ## Wikipedia example

# In[ ]:


question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 


# ## Python Agent

# In[ ]:


agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)


# In[ ]:


customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]


# In[ ]:


agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 


# #### View detailed outputs of the chains

# In[ ]:


import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False


# ## Define your own tool

# In[ ]:


#!pip install DateTime


# In[ ]:


from langchain.agents import tool
from datetime import date


# In[ ]:


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


# In[ ]:


agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)


# **Note**: 
# 
# The agent will sometimes come to the wrong conclusion (agents are a work in progress!). 
# 
# If it does, please try running it again.

# In[ ]:


try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")


# Reminder: Download your notebook to you local computer to save your work.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




