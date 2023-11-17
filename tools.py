from langchain.tools import BaseTool
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import openai
from dotenv import dotenv_values

### parameters #########
config = dotenv_values('conf.env')
openai.api_key = config['OPENAI_API_KEY']
SECURE_CONNECT_BUNDLE_PATH = config['SECURE_CONNECT_BUNDLE_PATH']
ASTRA_CLIENT_ID = config['ASTRA_CLIENT_ID']
ASTRA_CLIENT_SECRET = config['ASTRA_CLIENT_SECRET']
cloud_config = {
    'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
    }
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
 

class DataModelAgent(BaseTool):
    name = "DataModelAgent"
    description = "Use this tool as when the user wants help with data modelling. "

    def _run(self, user_question):
        KEYSPACE_NAME = config['ASTRA_KEYSPACE']
        TABLE_NAME = config['ASTRA_TABLE']
        model_id = "text-embedding-ada-002"
        embedding = openai.Embedding.create(openai_api_key=openai.api_key,input=user_question, model=model_id)['data'][0]['embedding']
        for row in session.execute(f"SELECT document_id,document,embedding_vector FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY embedding_vector ANN OF {embedding} LIMIT 1"):
                res = row.document 

        vector_search_result = res.replace("\n", " ")
        print('vector result: '+vector_search_result)

        # System messages and few-shot examples
        system_messages = [
            {'role':'system', 'content':'You are a chatbot for giving recommendations for the best practices in data modelling in Cassandra and you provide best practices recommendations and you provide a CQL with the best possible partition key for their table in the prompt.'},
            {'role':'system', 'content':'You need to ask questions related to how many rows do they expect in a partition and an initial draft for their table and you will calculate the partition size'},
            {'role':'system', 'content': "Here an example how to calculate the partition size: CREATE TABLE temperature_readings (device_id UUID,timestamp TIMESTAMP,temperature FLOAT,PRIMARY KEY (device_id, timestamp)); "},
            {'role':'system', 'content': f'They will have 500K rows for each device per day: Using the given data type size you can calculate the partition size:Partition size = (16+8+4)*500000 = 14000000 bytes = 13MB. This is a guiding example, adjust the calculation based on the actual data provided.'},
            {'role':'system', 'content': 'One of the best practices is to have a partition size less than 10MB. If it is above that, you can recommend using another column in the partition key or add time bucketing.'},
            {'role':'system', 'content': "Provide other best practices for Cassandra Data Modelling like using TTL, using a new denormalized table for each query access instead of using Secondary Index and Materialized Views"},
            {'role':'system', 'content': f'Result from vector search: {vector_search_result}'},
        ]

        # User message
        user_message = {'role': 'user', 'content': user_question}
        
        # Add all messages to the list
        messages_to_send = system_messages + [user_message]
        print('user message :'+str(messages_to_send))
        # Generate the response using the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages_to_send,
            temperature=0.5,  # Adjust temperature as needed
        )

        # Extract the content from the response
        chat_response = response.choices[0].message["content"]
        print('chat response: '+ str(chat_response))
        return chat_response

