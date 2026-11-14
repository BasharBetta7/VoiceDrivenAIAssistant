from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.messages import message_chunk_to_message
from langchain_core.tools import tool, Tool
from pydantic import BaseModel
from langchain.mcp import MCPAdapter
from langsmith import traceable

from dotenv import load_dotenv
import subprocess
import sys
from uuid import uuid4
import json
import asyncio
import os
import logging
import warnings
from langchain_core._api import LangChainBetaWarning
from src.tool_registry import ScriptRegistry, build_registry_tools
from langchain_core.prompts import ChatPromptTemplate
from src.events import *
from src.tools import *

from contextlib import asynccontextmanager




DEFAULTS = {'model':'qwen3.5:4b',
            'temperature':0.3,
            'max_tokens':2048,
            }

class LLMChat:
    def __init__(self, config:dict|None = None):
        if config:
            args = config
        else:
            args = DEFAULTS
        self.chat = ChatOllama(**args)
        self.sys_prompt = """You are a voice assistant with a persistent Python tool registry.
Search with search_scripts before creating a tool. Use execute_script with an
existing enabled tool's ID and schema-compatible arguments when it fits.
If no suitable tool exists, use write_script_tool to save a new script. Give
its entry-point function typed named parameters, a return annotation, a clear
docstring, and a JSON-serializable result. script_name has no path or extension.
New scripts are pending_validation: explain that review and enabling are needed
before execution. google_search tool is used when you need to access web data in case you are not sure of the answer to the query. Never claim a pending script has run. Tool failures are
returned as error objects; read them and correct the request when appropriate.
search_local_file is used to search for a specific filename and give the user its contents.
"""
        self.messages = [SystemMessage(content=self.sys_prompt)]
        self.registry = ScriptRegistry()
        self.tools = [*build_registry_tools(), google_search, read_local_file]
        self.available = None
        self.model_with_tools = None

    @asynccontextmanager
    async def session(self):
        mcp_config = {
                            "sequential-thinking": {
                                    "command": "npx",
                                    "args": [
                                    "-y",
                                    "@modelcontextprotocol/server-sequential-thinking"
                                    ]
                            },
                        }
        async with MCPAdapter(mcp_config) as adapter:
            mcp_tools = await adapter.list_tools()
            self.available = {
                tool.name: tool for tool in self.tools + mcp_tools
            }
            self.model_with_tools = self.chat.bind_tools(list(self.available.values()))
            try:
                yield self
            finally:
                self.available = None
                self.model_with_tools = None



    @traceable(name="llm_chat_no_tools", tags=["testing"])
    def llm_chat_no_tools(self, message:str, on_event:EventHandler | None = None) -> str:
        def emit(event:AgentEvent) -> None:
            if on_event is not None:
                on_event(event)
        self.messages.append(HumanMessage(content=message))
        emit({"type":"status", "data":{"text":"Thinking..."}})
        response = self.chat.invoke(self.messages)
        self.messages.append(response)
        if response.content:
            emit({
                "type":"message",
                "data":{
                    "content":response.content,
                    "final": not bool(response.tool_calls),
                }
            })

        return response.content

    @traceable(name="voice_agent")
    async def llm_chat(self, message: str, max_rounds: int = 8, on_event: EventHandler | None = None) -> list:
        if self.available is None or self.model_with_tools is None:
            raise RuntimeError("No activate session was opened. Make sure to run the agent inside 'async with agent.session()'")

        def emit(event:AgentEvent) -> None:
            if on_event is not None:
                on_event(event)

        available = self.available
        model_with_tools = self.model_with_tools
        self.messages.append(HumanMessage(content=message))
        for _ in range(max_rounds):
            emit({"type":"status", "data":{"text":"Thinking..."}})
            combined = None
            async for chunk in model_with_tools.astream(self.messages):
                combined = chunk if combined is None else combined + chunk
                if chunk.text:
                    emit({"type": "message_delta", "data": {"text": chunk.text}})
            if combined is None:
                raise RuntimeError("The model returned an empty stream.")
            # Tool arguments may span chunks. Execute only after assembly.
            result = message_chunk_to_message(combined)
            self.messages.append(result)
            if result.content:
                logging.info("[AGENT]: %s", result.content)
                emit({
                    "type":"message",
                    "data":{
                        "content":result.content,
                        "final": not bool(result.tool_calls),
                        "streamed": True,
                    }
                })
            if not result.tool_calls:
                return self.messages
            for call in result.tool_calls:
                emit({
                    "type":"status",
                    "data":{
                        "text":f"Running {call['name']}..."
                    }
                })
                logging.info(f"[TOOL CALL]: {call['name']}")
                try:
                    registered_tool = available[call["name"]]
                    # emit({
                    #     "type":"tool_started",
                    #     "data":{
                    #         "name":call["name"],
                    #     }
                    # })
                    tool_result = await registered_tool.ainvoke(call["args"])
                    # emit({
                    #     "type":"tool_result",
                    #     "data":{
                    #         "result":f"{tool_result}"
                    #     }
                    # })
                except Exception as exc:
                    tool_result = {"error": str(exc)}
                logging.info(f"[OBS]: {tool_result}")
                self.messages.append(ToolMessage(
                    content=json.dumps(tool_result), tool_call_id=call["id"]
                ))
        raise RuntimeError("Agent reached its tool-call round limit.")
    def _get_messages(self):
        return self.messages.copy()

    def flush_messages(self):
        "Reset session, keeping system prompt"
        self.messages = self.messages[:1]
    def __str__(self):
        return "\n\n".join([f"-- [{m.type.upper()}]: {m.content}" for m in self._get_messages() ])






if __name__ == '__main__':
    load_dotenv()
    os.environ['LANGSMITH_TRACING'] = "true"
    chat = LLMChat()
    result = chat.llm_chat_no_tools('What is the square root of nine?')
    print(result)
    print(chat._get_messages())
