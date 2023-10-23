import os
import aiohttp
import chainlit as cl
from aiohttp import BasicAuth
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor

load_dotenv()

API_URL = os.getenv("API_URL")
CREDS = BasicAuth(os.getenv("AUTH_USERNAME"), os.getenv("AUTH_PASSWORD"))


async def get_assortment(*args, **kwargs):
    async with aiohttp.ClientSession(auth=CREDS) as session:
        async with session.get(f"{API_URL}/assortment") as response:
            response.raise_for_status()
            res = await response.json()
            return [{"name": i["name"]} for i in res["rows"]]


async def get_stores(*args, **kwargs):
    async with aiohttp.ClientSession(auth=CREDS) as session:
        async with session.get(f"{API_URL}/store") as response:
            response.raise_for_status()
            res = await response.json()
            return [{"id": i["id"], "name":i["name"], "address":i["address"]} for i in res["rows"]]

get_assortment_tool = Tool.from_function(
    func=get_assortment,
    name="Get assortment",
    description="useful for when you need to answer questions about assortment",
    coroutine=get_assortment
)

get_store_tool = Tool.from_function(
    func=get_stores,
    name="Get stores",
    description="useful for when you need to answer questions about stores, location of outlets and where you can buy goods",
    coroutine=get_stores
)


llm = OpenAI(openai_api_key=os.getenv("OPENAI_KEY"))


@cl.on_chat_start
def main():
    agent = initialize_agent(llm=llm, tools=[
        get_assortment_tool, get_store_tool], verbose=True, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")
    res = await agent.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["output"]).send()
