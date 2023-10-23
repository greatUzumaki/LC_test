import os

import aiohttp
import chainlit as cl
from aiohttp import BasicAuth
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field

set_llm_cache(InMemoryCache())


class Schema(BaseModel):
    variant: str = Field()


load_dotenv()

API_URL = os.getenv("API_URL")
CREDS = BasicAuth(os.getenv("AUTH_USERNAME"), os.getenv("AUTH_PASSWORD"))


async def get_assortment(pathname: str, *args, **kwargs):
    async with aiohttp.ClientSession(auth=CREDS) as session:
        async with session.get(f"{API_URL}/assortment?filter=pathname~{pathname}&limit=10") as response:
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
    description="This information is helpful for responding to queries about different product categories. It can recognize and categorize user queries into one of the following product types: POD (for questions about POD systems), Щелочные (for inquiries regarding e-liquids), and Одноразовые (for addressing disposable vapes).",
    coroutine=get_assortment,
    args_schema=Schema
)

get_store_tool = Tool.from_function(
    func=get_stores,
    name="Get stores",
    description="useful for when you need to answer questions about stores, location of outlets and where you can buy goods",
    coroutine=get_stores
)


@cl.on_chat_start
def main():
    agent_kwargs = {
        "system_message": SystemMessage(content="You are a sales consultant in a tobacco store, answer only in Russian"),
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_KEY"))

    agent = initialize_agent(llm=llm, tools=[get_assortment_tool, get_store_tool], agent_kwargs=agent_kwargs, verbose=True,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, handle_parsing_errors=True)
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")
    res = await agent.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["output"]).send()
