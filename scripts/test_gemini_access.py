import os
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()

for key, value in os.environ.items():
    print(f"{key}={value}")

client = Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents='What is the answer to the Ultimate Question of Life, the Universe, and Everything'
)

response_2 = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What was my previous question? History of this chat: 'What is the answer to the Ultimate Question of Life, the Universe, and Everything'")

def print_response(resp):
    if resp.text:
        print(resp.text)


print_response(response)

print_response(response_2)


# Close the sync client to release resources.
client.close()


