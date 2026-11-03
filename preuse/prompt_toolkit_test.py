from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
import sys
import asyncio
import os
os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

from src.llm_model import LLMChat
from dotenv import load_dotenv
from src.events import *
STATUS_MSG_STYLE = "dim"
CONTENT_MSG_STYLE = "blue"
console = Console(file=sys.stdout)
status = console.status("Thinking...",)

stream_display = None
stream_text = ""


def finish_stream() -> None:
    global stream_display, stream_text
    if stream_display is not None:
        stream_display.stop()
        stream_display = None
    stream_text = ""


def render_event(event: AgentEvent) -> None:
    global stream_display, stream_text
    data = event["data"]
    if event["type"] == "message_delta":
        status.stop()
        stream_text += data["text"]
        if stream_display is None:
            stream_display = Live(
                console=console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            stream_display.start()
        stream_display.update(
            Markdown(stream_text, style=CONTENT_MSG_STYLE), refresh=True
        )
        return

    if event["type"] == "status":
        finish_stream()
        status.update(data["text"])
        status.start()
    else:
        status.stop()

        if event["type"] == "message":
            if data.get("streamed") and stream_display is not None:
                finish_stream()
                return
            content = data["content"]
            if isinstance(content, str):
                console.print(Markdown(content), style=CONTENT_MSG_STYLE)
            else:
                console.print(content, style=CONTENT_MSG_STYLE)
        elif event["type"] == "tool_started":
            finish_stream()
            console.print(
                f"Running {data['name']}...",
                style='cyan',
                markup=False,
            )

        elif event["type"] == "tool_result":
            console.print(data['result'],
                          style='cyan',
                          markup=False,)




async def main():
    load_dotenv()
    agent = LLMChat()

    session = PromptSession(
        completer=WordCompleter(["/help","/new","/quit", "/chathistory"]),
    )
    console.print("[bold cyan]Voice Assistant[/bold cyan]")
    console.print("/help for commands . Ctrl-D to exit")
    async with agent.session():
        with patch_stdout(raw=True):
            while True:
                try:
                    text = (await session.prompt_async("You > ")).strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue

                if not text:
                    continue
                if text == "/quit":
                    break
                if text == "/help":
                    console.print("/new - reset conversation.\n/quit - exit.\n/chathistory - get session history.\n")
                    continue
                if text == "/new":
                    agent.flush_messages()
                    console.print("Started a new conversation.")
                    continue
                if text == "/chathistory":
                    console.print(Markdown(str(agent)))
                    continue
                if text.startswith("/"):
                    console.print("Unkown command. Use /help.")
                    continue
                try:
                    await agent.llm_chat(message=text, on_event=render_event)
                except Exception as e:
                    finish_stream()
                    console.print("Request Failed:", str(e), markup=False)
                finally:
                    status.stop()
                    finish_stream()





if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
