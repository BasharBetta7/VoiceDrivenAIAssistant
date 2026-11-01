from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
import asyncio
import os 
os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

from src.llm_model import LLMChat
from dotenv import load_dotenv

async def main():
    load_dotenv()
    agent = LLMChat()
    console = Console()
    session = PromptSession(
        completer=WordCompleter(["/help","/new","/quit", "/chathistory"]),
    )
    console.print("[bold cyan]Voice Assistant[/bold cyan]")
    console.print("/help for commands . Ctrl-D to exit")
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
                agent.llm_chat_no_tools(text)
                console.print(Markdown(f"{agent.messages[-1].content}"),style="blue")
            except Exception as e:
                console.print("Request Failed:", str(e), markup=False)

            



if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())