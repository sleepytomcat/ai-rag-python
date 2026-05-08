import time

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool


def main():
    load_dotenv()

    conversation = [
        SystemMessage("You are a helpful weather agent who also adds a joke to replies."),
        HumanMessage("What is the weather now in San Francisco?"),
        AIMessage("As of right now in San Francisco, the weather is partly cloudy with a temperature of around 62°F."),
        HumanMessage("Is it warmer in Vienna now?")
    ]

    chat_model = init_chat_model(
        model="gemini-3.1-flash-lite",
        model_provider="google_genai",
        temperature=0.1
    )

    t0 = time.perf_counter()
    for chunk in chat_model.stream(conversation):
        print(chunk.text, end="[CHUNK END]", flush=True)
    print(f"chat_model.invoke(conversation) took {time.perf_counter() - t0:.2f}s")

@tool('get_weather', description='get weather in a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.text


if __name__ == "__main__":
    main()
