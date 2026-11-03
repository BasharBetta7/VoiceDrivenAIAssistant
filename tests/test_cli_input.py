"""Exercise terminal input routing without models or microphone access."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console


def load_cli():
    path = Path(__file__).parents[1] / "preuse" / "prompt_toolkit_test.py"
    spec = importlib.util.spec_from_file_location("cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.console = Console(file=io.StringIO())
    module.status = module.console.status("")
    return module


class CLIInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_f2_returns_voice_request_and_preserves_typed_buffer(self):
        cli = load_cli()
        with create_pipe_input() as pipe:
            session = PromptSession(input=pipe, output=DummyOutput(), key_bindings=cli.input_bindings())
            pipe.send_text("unfinished\x1bOQ")
            result = await asyncio.wait_for(session.prompt_async(), timeout=3)
        self.assertIs(result, cli.VOICE_INPUT)
        self.assertEqual(session.default_buffer.text, "unfinished")

    async def run_cli(self, voice_results):
        cli = load_cli()
        @asynccontextmanager
        async def agent_session():
            yield

        agent = SimpleNamespace(session=agent_session, llm_chat=AsyncMock())
        voice = SimpleNamespace(transcribe_once=AsyncMock(side_effect=voice_results))
        factory = Mock(return_value=voice)
        pipeline = ModuleType("src.pipeline")
        pipeline.VoiceAssistant = factory
        draft = Document("unfinished", cursor_position=4)
        session = SimpleNamespace(
            prompt_async=AsyncMock(side_effect=["typed prompt"] + [cli.VOICE_INPUT] * len(voice_results) + ["/quit"]),
            default_buffer=SimpleNamespace(document=draft),
        )
        with (
            patch.object(cli, "LLMChat", return_value=agent),
            patch.object(cli, "PromptSession", return_value=session),
            patch.object(cli, "patch_stdout", return_value=nullcontext()),
            patch.dict(sys.modules, {"src.pipeline": pipeline}),
        ):
            await cli.main()
        return cli, agent, factory, session, draft

    async def test_voice_and_text_share_agent_and_restore_draft(self):
        cli, agent, factory, session, draft = await self.run_cli(["spoken prompt", "another spoken prompt"])
        self.assertEqual([call.kwargs["message"] for call in agent.llm_chat.await_args_list],
                         ["typed prompt", "spoken prompt", "another spoken prompt"])
        factory.assert_called_once()
        self.assertIs(factory.call_args.kwargs["llm"], agent)
        self.assertIs(session.prompt_async.await_args_list[-1].kwargs["default"], draft)

    async def test_typing_does_not_initialize_voice_models(self):
        _, agent, factory, _, _ = await self.run_cli([])
        factory.assert_not_called()
        agent.llm_chat.assert_awaited_once()

    async def test_failed_or_rejected_voice_input_returns_to_text_prompt(self):
        _, agent, _, _, _ = await self.run_cli([RuntimeError("microphone unavailable"), None])
        agent.llm_chat.assert_awaited_once()
        self.assertEqual(agent.llm_chat.await_args.kwargs["message"], "typed prompt")


if __name__ == "__main__":
    unittest.main()
