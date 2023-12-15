import streamlit as st
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import dotenv_values
from dotenv import load_dotenv
import os
from tools import DataModelAgent,CassandraAgent,AstraMigrationAgent
from langchain.schema import SystemMessage
##############################
### initialize agent #########
##############################
tools = [DataModelAgent(),CassandraAgent(),AstraMigrationAgent()]
config = dotenv_values('conf.env')
load_dotenv('conf.env')
trace= os.getenv('LANGCHAIN_TRACING_V2')
langsmith_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langsmith_api = os.getenv('LANGCHAIN_API_KEY')
print("Trace on/off : " + trace)
openai_key = config['OPENAI_API_KEY']
ASTRA_DB_KEYSPACE = config['ASTRA_KEYSPACE']


conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0,
    model_name="gpt-4"
)

system_message = SystemMessage(content="You are a Data Model expert in Cassandra Data Modelling")

agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=False,
    memory=conversational_memory,
    handle_parsing_errors=True,
    early_stopping_method='generate',
    max_execution_time=1,
     agent_kwargs={
        "system_message": system_message.content
    }
)


user_question = st.text_input('Can you give an explanation about your data model and initial table design please?')

# write agent response
if user_question and user_question != "":
     with st.spinner(text="In progress..."):
        response = agent.run('{}'.format(user_question))
        st.write(response)

