from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool, Tool
from pydantic import BaseModel
from langchain.mcp import MCPAdapter


import subprocess
import sys 
from uuid import uuid4
import json
import asyncio

from src.tool_registry import ScriptRegistry, build_registry_tools


DEFAULTS = {'model':'qwen3.5:4b',
            'temperature':0,
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
before execution. Never claim a pending script has run. Tool failures are
returned as error objects; read them and correct the request when appropriate.
"""
        self.messages = [SystemMessage(content=self.sys_prompt)]
        self.registry = ScriptRegistry()
        self.tools = build_registry_tools(self.registry)
    async def llm_chat(self, message: str, max_rounds: int = 8) -> list:
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
            available = {registered_tool.name: registered_tool for registered_tool in self.tools + mcp_tools}
            model_with_tools = self.chat.bind_tools(list(available.values()))
            self.messages.append(HumanMessage(content=message))
            for _ in range(max_rounds):
                result = await model_with_tools.ainvoke(self.messages)
                self.messages.append(result)
                if result.content:
                    print(f"[AGENT]: {result.content}")
                if not result.tool_calls:
                    return self.messages
                for call in result.tool_calls:
                    print(f"[TOOL CALL]: {call['name']}")
                    try:
                        registered_tool = available[call["name"]]
                        tool_result = await registered_tool.ainvoke(call["args"])
                    except Exception as exc:
                        tool_result = {"error": str(exc)}
                    print(f"[OBS]: {tool_result}")
                    self.messages.append(ToolMessage(
                        content=json.dumps(tool_result), tool_call_id=call["id"]
                    ))
            raise RuntimeError("Agent reached its tool-call round limit.")
    def _get_messages(self):
        return self.messages.copy()



        

if __name__ == '__main__':
    chat = LLMChat()
    result = asyncio.run(chat.llm_chat('Hello, what is the age of the universe?'))
    print(result)
    print(chat._get_messages())
