import time

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


def main():
    load_dotenv()

    print("> Chat model call")
    chat_model = init_chat_model(
        model="gemini-3.1-flash-lite",
        model_provider="google_genai",
        temperature=0.1
    )

    t0 = time.perf_counter()
    response = chat_model.invoke("What is the weather in San Francisco now? Would you recommend to have an umbrella?")
    print(f"chat_model.invoke() took {time.perf_counter() - t0:.2f}s")

    print(response.text)

    print("> Agent call")
    agent = create_agent(model=chat_model, tools=[get_weather], system_prompt="You are a helpful weather agent who also adds a dark joke to replies.")

    t0 = time.perf_counter()
    response = agent.invoke(
        {
            'messages': [
                {"role":"user", "content": "What is the weather in Vienna now?"}
            ]
        }
    )
    print(f"agent.invoke() took {time.perf_counter() - t0:.2f}s")

    print(response['messages'][-1])

@tool('get_weather', description='get weather in a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.text


if __name__ == "__main__":
    main()
