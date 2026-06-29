import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from typing import TypedDict


class TravelState(TypedDict):
    country: str
    capital: str
    temperature_celsius: float
    response: str


def main():
    load_dotenv()

    chat_model = init_chat_model(
        model="gemini-3-flash-preview",
        model_provider="google_genai",
        temperature=0.7,
    )

    def find_capital(state: TravelState) -> dict:
        msg = HumanMessage(
            content=f"What is the capital city of {state['country']}? Reply with just the city name, nothing else."
        )
        result = chat_model.invoke([msg])
        content = result.content
        text = content if isinstance(content, str) else content[0].get("text", "")
        return {"capital": text.strip()}

    def find_temperature(state: TravelState) -> dict:
        data = requests.get(
            f"https://wttr.in/{state['capital']}?format=j1"
        ).json()
        temp_c = float(data["current_condition"][0]["temp_C"])
        return {"temperature_celsius": temp_c}

    def generate_response(state: TravelState) -> dict:
        msg = HumanMessage(content=(
            f"Write a friendly two-sentence response about {state['country']}. "
            f"Mention that its capital is {state['capital']} and the current "
            f"temperature there is {state['temperature_celsius']}°C."
        ))
        result = chat_model.invoke([msg])
        content = result.content
        text = content if isinstance(content, str) else content[0].get("text", "")
        return {"response": text}

    checkpointer = InMemorySaver()

    builder = StateGraph(TravelState)
    builder.add_node("find_capital", find_capital)
    builder.add_node("find_temperature", find_temperature)
    builder.add_node("generate_response", generate_response)
    builder.set_entry_point("find_capital")
    builder.add_edge("find_capital", "find_temperature")
    builder.add_edge("find_temperature", "generate_response")
    builder.add_edge("generate_response", END)
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "demo-thread"}}

    result = graph.invoke({"country": "France"}, config=config)
    print(result["response"])
    print()

    result2 = graph.invoke({"country": "Japan"}, config=config)
    print(result2["response"])
    print()

    print("--- Checkpoints ---")
    for checkpoint in checkpointer.list(config):
        print(checkpoint)


if __name__ == "__main__":
    main()
