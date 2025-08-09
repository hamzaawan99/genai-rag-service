import os
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=9900,
    grpc_port=50051,
    auth_credentials=Auth.api_key(weaviate_api_key)
)

print(client.is_ready())

assert client.is_ready()

client.close()