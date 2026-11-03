"""Check streaming and fragmented tool calls without Ollama or MCP servers."""

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, ToolMessage
from rich.console import Console

from src.llm_model import LLMChat


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def run_chat(self, rounds, tools=None):
        events = []
        agent = LLMChat.__new__(LLMChat)
        agent.messages = [SystemMessage(content="Test assistant")]
        agent.available = tools or {}
        remaining = iter(rounds)

        async def stream(messages):
            for chunk in next(remaining):
                yield chunk
                # Deltas must be delivered before the stream finishes.
                if chunk.text:
                    self.assertEqual(events[-1]["type"], "message_delta")
                    self.assertEqual(events[-1]["data"]["text"], chunk.text)

        agent.model_with_tools = SimpleNamespace(astream=stream)
        # Test the agent loop independently of LangSmith's tracing transport.
        await LLMChat.llm_chat.__wrapped__(agent, "hello", on_event=events.append)
        return agent, events

    async def test_streamed_answer_is_saved_once_as_complete_message(self):
        agent, events = await self.run_chat([[
            AIMessageChunk(content="Hello "), AIMessageChunk(content="world"),
        ]])
        self.assertEqual(agent.messages[-1].content, "Hello world")
        self.assertIs(type(agent.messages[-1]), AIMessage)
        self.assertEqual(len(agent.messages), 3)
        self.assertTrue(events[-1]["data"]["streamed"])

    async def test_fragmented_tool_arguments_are_assembled_before_execution(self):
        tool = SimpleNamespace(ainvoke=AsyncMock(return_value=5))
        agent, events = await self.run_chat([
            [
                AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "add", "args": '{"a":', "id": "call-1", "index": 0,
                }]),
                AIMessageChunk(content="", tool_call_chunks=[{
                    "name": None, "args": ' 2, "b": 3}', "id": None, "index": 0,
                }]),
            ],
            [AIMessageChunk(content="The result is 5.")],
        ], {"add": tool})
        tool.ainvoke.assert_awaited_once_with({"a": 2, "b": 3})
        self.assertIsInstance(agent.messages[-2], ToolMessage)
        self.assertEqual(agent.messages[-2].tool_call_id, "call-1")
        self.assertEqual(agent.messages[-1].content, "The result is 5.")
        self.assertIn("tool_started", [event["type"] for event in events])

    async def test_empty_stream_reports_failure(self):
        with self.assertRaisesRegex(RuntimeError, "empty stream"):
            await self.run_chat([[]])


class StreamingRendererTests(unittest.TestCase):
    def test_final_event_does_not_duplicate_streamed_markdown(self):
        path = Path(__file__).parents[1] / "preuse" / "prompt_toolkit_test.py"
        spec = importlib.util.spec_from_file_location("streaming_cli_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = io.StringIO()
        module.console = Console(file=output, force_terminal=False, width=80)
        module.status = module.console.status("Thinking...")
        module.render_event({"type": "message_delta", "data": {"text": "Hello "}})
        module.render_event({"type": "message_delta", "data": {"text": "world"}})
        module.render_event({"type": "message", "data": {
            "content": "Hello world", "streamed": True, "final": True,
        }})
        self.assertEqual(output.getvalue().count("Hello world"), 1)
        self.assertIsNone(module.stream_display)


if __name__ == "__main__":
    unittest.main()
