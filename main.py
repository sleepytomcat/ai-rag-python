import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    city_name: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


def main():
    load_dotenv()

    chat_model = init_chat_model(
        model="gemini-3.1-flash-lite",
        model_provider="google_genai",
        temperature=0.1
    )

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=chat_model,
        tools=[get_weather, locate_user],
        system_prompt="You are a helpful weather agent who also adds a joke to the replies.",
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread_id": "123"
        }
    }

    t0 = time.perf_counter()
    response = agent.invoke(
        {
            'messages': [
                {
                    "role": "user",
                    "content": "What is the weather in my location?"
                }
            ],
        },
        config=config,
        context=Context(user_id="John")
    )

    print(f"agent.invoke() took {time.perf_counter() - t0:.2f}s")

    print(response['structured_response'])

    response2 = agent.invoke(
        {
            'messages': [
                {
                    "role": "user",
                    "content": "And is this usual?"
                }
            ],
        },
        config=config,
        context=Context(user_id="John")
    )

    print(response2['structured_response'])


@tool('get_weather', description='get weather in a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.text


@tool('locate_user', description="get user's city name based on context", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]) -> str:
    match runtime.context.user_id:
        case "John":
            return "Vienna"
        case "Jim":
            return "San Francisco"
        case _:
            return "Unknown"


if __name__ == "__main__":
    main()
