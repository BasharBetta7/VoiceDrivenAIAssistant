from langchain_core.tools import tool
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import tempfile
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState
from typing import Optional


from dotenv import load_dotenv


@tool
def google_search(query:str) -> dict:
    """Google search tool"""
    results = GoogleSerperAPIWrapper(k=5).results(query)
    return results


def build_chain():
    prompt_template = PromptTemplate.from_template("""Search the web for the following topic, and proive your answer.
                                                   topic: {topic}""")
    class WebResultTemplate(BaseModel):
        result: str = Field(..., description="The result of the web page")
        link: Optional[str] = Field(description="the link to the web page")

    model = ChatOllama(model='qwen3.5:4b')
    agent = create_agent(
        model=model,
        tools=[google_search],
        system_prompt="Use search when you don't know the answer.",
    )
    chain = prompt_template | model
    res = agent.invoke({"messages":[{"role":"user","content":"What are good phones below 500 dollars? in 2026"}]})
    print(res['messages'][-1].content)

    

if __name__ == '__main__':
    load_dotenv()
    build_chain()
    
    