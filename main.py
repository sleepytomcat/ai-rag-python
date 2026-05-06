from datasets import load_from_disk
from dotenv import load_dotenv
import requests
from langchain.agents import create_agent
from langchain.tools import tool



def main():
    load_dotenv()
    print(get_weather("San Francisco"))
    pass


#@tool('get_weather', description='get weather in a given city', return_direct=True)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}')
    return response.text


if __name__ == "__main__":
    main()
