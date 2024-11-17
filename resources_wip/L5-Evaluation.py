#!/usr/bin/env python
# coding: utf-8

# # LangChain: Evaluation
# 
# ## Outline:
# 
# * Example generation
# * Manual evaluation (and debuging)
# * LLM-assisted evaluation
# * LangChain evaluation platform

# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


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


# ## Create our QandA application

# In[ ]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch


# In[ ]:


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()


# In[ ]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


# In[ ]:


llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)


# ### Coming up with test datapoints

# In[ ]:


data[10]


# In[ ]:


data[11]


# ### Hard-coded examples

# In[ ]:


examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]


# ### LLM-Generated examples

# In[ ]:


from langchain.evaluation.qa import QAGenerateChain


# In[ ]:


example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))


# In[ ]:


# the warning below can be safely ignored


# In[ ]:


new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)


# In[ ]:


new_examples[0]


# In[ ]:


data[0]


# ### Combine examples

# In[ ]:


examples += new_examples


# In[ ]:


qa.run(examples[0]["query"])


# ## Manual Evaluation

# In[ ]:


import langchain
langchain.debug = True


# In[ ]:


qa.run(examples[0]["query"])


# In[ ]:


# Turn off the debug mode
langchain.debug = False


# ## LLM assisted evaluation

# In[ ]:


predictions = qa.apply(examples)


# In[ ]:


from langchain.evaluation.qa import QAEvalChain


# In[ ]:


llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)


# In[ ]:


graded_outputs = eval_chain.evaluate(examples, predictions)


# In[ ]:


for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()


# In[ ]:


graded_outputs[0]


# ## LangChain evaluation platform

# The LangChain evaluation platform, LangChain Plus, can be accessed here https://www.langchain.plus/.  
# Use the invite code `lang_learners_2023`

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




